#include "pch.h"
#include "PPOcr.hpp"

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Graphics::Imaging;

static bool g_DxGenericMLSupport{};
static bool g_DxUseCore{};

// Source: https://github.com/microsoft/DirectML/blob/master/Samples/DirectMLNpuInference/main.cpp
bool TryGetProperty(IDXCoreAdapter* adapter, DXCoreAdapterProperty prop, std::string& outputValue) {
    if (adapter->IsPropertySupported(prop)) {
        size_t propSize;
        check_hresult(adapter->GetPropertySize(prop, &propSize));

        outputValue.resize(propSize);
        check_hresult(adapter->GetProperty(prop, propSize, outputValue.data()));

        // Trim any trailing nul characters. 
        while (!outputValue.empty() && outputValue.back() == '\0') {
            outputValue.pop_back();
        }

        return true;
    }
    return false;
}

std::string GetProperty(IDXCoreAdapter* adapter, DXCoreAdapterProperty prop) {
    std::string value;
    if (!TryGetProperty(adapter, prop, value)) {
        throw hresult_error(E_FAIL, L"Failed to retrieve the requested property.");
    }
    return value;
}

template <typename F>
void ReadFileByLinesWinAPI(hstring const& filename, F&& f) {
    file_handle hFile{ CreateFileW(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr) };
    if (!hFile) {
        throw_last_error();
    }

    const DWORD bufferSize = 4096;
    char buffer[bufferSize];
    DWORD bytesRead = 0;
    std::string leftover;

    while (ReadFile(hFile.get(), buffer, bufferSize, &bytesRead, nullptr) && bytesRead > 0) {
        size_t start = 0;
        for (DWORD i = 0; i < bytesRead; i++) {
            if (buffer[i] == '\n') {
                size_t len = i - start;
                // Handle \r\n
                if (len > 0 && buffer[start + len - 1] == '\r') { --len; }
                std::string line = std::move(leftover);
                line += std::string_view(&buffer[start], len);
                f(std::move(line));
                leftover.clear();
                start = i + 1;
            }
        }
        // Save any partial line for next read
        if (start < bytesRead) {
            leftover += std::string_view(&buffer[start], bytesRead - start);
        }
    }
    // Handle last line if file doesn't end with newline
    if (!leftover.empty()) {
        f(std::move(leftover));
    }
}

/// <summary>
/// Enumerates D3D12 devices using DXGI.
/// </summary>
/// <param name="filter">The filter functor with signature `fn(IDXGIAdapter1*) -> bool`, returning
/// whether iteration should continue.</param>
template <typename F>
void EnumerateD3D12DevicesViaDxgiWithFilter(F&& filter) {
    com_ptr<IDXGIFactory1> dxgiFactory;
    check_hresult(CreateDXGIFactory1(guid_of<IDXGIFactory1>(), dxgiFactory.put_void()));
    com_ptr<IDXGIAdapter1> adapter;
    for (UINT i = 0; dxgiFactory->EnumAdapters1(i, adapter.put()) != DXGI_ERROR_NOT_FOUND; i++) {
        if (!filter(adapter.get())) {
            break; // Stop iteration if the filter returns false
        }
    }
}

/// <summary>
/// Enumerates D3D12 devices using DxCore.
/// </summary>
/// <param name="filter">The filter functor with signature `fn(IDXCoreAdapter*) -> bool`, returning
/// whether iteration should continue.</param>
template <typename F>
void EnumerateD3D12DevicesViaDxCoreWithFilter(F&& filter) {
    com_ptr<IDXCoreAdapterFactory> factory;
    check_hresult(DXCoreCreateAdapterFactory(guid_of<decltype(factory)>(), factory.put_void()));
    com_ptr<IDXCoreAdapterList> adapterList;
    auto& attrPtr = g_DxGenericMLSupport ? DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML : DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE;
    check_hresult(factory->CreateAdapterList(1, &attrPtr, adapterList.put()));

    auto adapterCount = adapterList->GetAdapterCount();
    for (uint32_t i = 0; i < adapterCount; ++i) {
        com_ptr<IDXCoreAdapter> adapter;
        check_hresult(adapterList->GetAdapter(i, adapter.put()));
        if (!filter(adapter.get())) {
            break; // Stop iteration if the filter returns false
        }
    }
}

/// <summary>
/// Selects a D3D12 device suitable for machine learning tasks.
/// </summary>
/// <param name="device">The preferred device name. Can be empty.</param>
/// <returns>The D3D12 device.</returns>
com_ptr<ID3D12Device> PickD3D12DeviceForML(hstring const& device) {
    D3D_FEATURE_LEVEL featureLevel = g_DxGenericMLSupport ? D3D_FEATURE_LEVEL_1_0_GENERIC : D3D_FEATURE_LEVEL_1_0_CORE;

    com_ptr<ID3D12Device> d3dDevice{ nullptr };
    if (device == L"CPU") {
        d3dDevice = nullptr; // No device needed for CPU inference
    }
    else if (!device.empty()) {
        // If a specific device is requested, try to find it
        if (g_DxUseCore) {
            EnumerateD3D12DevicesViaDxCoreWithFilter([&, device = to_string(device)](IDXCoreAdapter* adapter) {
                std::string description = GetProperty(adapter, DXCoreAdapterProperty::DriverDescription);
                if (description.starts_with(device)) {
                    d3dDevice = capture<ID3D12Device>(D3D12CreateDevice, adapter, featureLevel);
                    return false; // Stop enumeration
                }
                return true; // Continue enumeration
            });
        }
        else {
            EnumerateD3D12DevicesViaDxgiWithFilter([&](IDXGIAdapter1* adapter) {
                DXGI_ADAPTER_DESC1 desc;
                check_hresult(adapter->GetDesc1(&desc));
                if (std::wstring_view(desc.Description).starts_with(device)) {
                    d3dDevice = capture<ID3D12Device>(D3D12CreateDevice, adapter, featureLevel);
                    return false; // Stop enumeration
                }
                return true; // Continue enumeration
            });
        }

        // If no specific device is found or requested, bail
        if (!d3dDevice) {
            throw hresult_error(E_FAIL, L"Failed to create the requested D3D12 device.");
        }
    }
    else {
        d3dDevice = capture<ID3D12Device>(D3D12CreateDevice, nullptr, featureLevel);
    }
    return d3dDevice;
}

LearningModelDevice CreateLearningModelDevice(com_ptr<ID3D12Device> const& d3dDevice) {
    if (!d3dDevice) {
        // If no device is provided, use the CPU for inference
        return LearningModelDevice(LearningModelDeviceKind::Cpu);
    }

    // Create a LearningModelDevice from the D3D12 device
    auto factory = get_activation_factory<ILearningModelDeviceFactoryNative>(name_of<LearningModelDevice>());

    D3D12_COMMAND_QUEUE_DESC queueDesc{
        .Type = D3D12_COMMAND_LIST_TYPE_COMPUTE,
        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
    };
    auto commandQueue = capture<ID3D12CommandQueue>(d3dDevice, &ID3D12Device::CreateCommandQueue, &queueDesc);

    LearningModelDevice device{ nullptr };
    check_hresult(factory->CreateFromD3D12CommandQueue(
        commandQueue.get(),
        reinterpret_cast<::IUnknown**>(put_abi(device))
    ));

    return device;
}

std::vector<hstring> EnumerateD3D12DevicesViaDxgi() {
    std::vector<hstring> devices;
    EnumerateD3D12DevicesViaDxgiWithFilter([&devices](IDXGIAdapter1* adapter) {
        DXGI_ADAPTER_DESC1 desc;
        check_hresult(adapter->GetDesc1(&desc));
        devices.push_back(to_hstring(desc.Description));
        return true; // Continue enumeration
    });
    return devices;
}

std::vector<hstring> EnumerateD3D12DevicesViaDxCore() {
    std::vector<hstring> devices;
    EnumerateD3D12DevicesViaDxCoreWithFilter([&devices](IDXCoreAdapter* adapter) {
        std::string description = GetProperty(adapter, DXCoreAdapterProperty::DriverDescription);
        devices.push_back(to_hstring(description));
        return true; // Continue enumeration
    });
    return devices;
}

template <typename T>
bool VectorEquals(IVectorView<T> const& vec, std::initializer_list<std::type_identity_t<T>> list) {
    auto ib = vec.begin();
    auto ie = vec.end();
    return std::equal(ib, ie, list.begin(), list.end());
}

com_ptr<IWICBitmap> SoftwareBitmapToWICBitmap(SoftwareBitmap const& bitmap) {
    auto result = try_capture<IWICBitmap>(bitmap.as<ISoftwareBitmapNative>(), &ISoftwareBitmapNative::GetData);
    if (!result) {
        auto converted = SoftwareBitmap::Convert(bitmap, BitmapPixelFormat::Bgra8);
        result = capture<IWICBitmap>(converted.as<ISoftwareBitmapNative>(), &ISoftwareBitmapNative::GetData);
    }
    return result;
}

struct WICBitmapSize {
    uint32_t width, height;
};

WICBitmapSize GetWICBitmapSize(com_ptr<IWICBitmap> const& bmp) {
    UINT w, h;
    check_hresult(bmp->GetSize(&w, &h));
    return { w, h };
}

com_ptr<IWICBitmap> ResizeWICBitmap(com_ptr<IWICBitmap> const& bmp, uint32_t width, uint32_t height) {
    static auto factory = create_instance<IWICImagingFactory>(CLSID_WICImagingFactory, CLSCTX_INPROC_SERVER);
    com_ptr<IWICBitmapScaler> scaler;
    check_hresult(factory->CreateBitmapScaler(scaler.put()));
    // Initialize the scaler with the bitmap and the new size
    //check_hresult(scaler->Initialize(bmp.get(), width, height, WICBitmapInterpolationMode::WICBitmapInterpolationModeFant));
    check_hresult(scaler->Initialize(bmp.get(), width, height, WICBitmapInterpolationMode::WICBitmapInterpolationModeLinear));
    // Create a new bitmap to hold the resized image
    com_ptr<IWICBitmap> resizedBmp;
    check_hresult(factory->CreateBitmapFromSource(scaler.get(), WICBitmapCreateCacheOption::WICBitmapNoCache, resizedBmp.put()));
    return resizedBmp;
}

com_ptr<IWICBitmap> ResizeWICBitmapByHeight(com_ptr<IWICBitmap> const& bmp, uint32_t newHeight) {
    auto [w, h] = GetWICBitmapSize(bmp);
    double ratio = static_cast<double>(newHeight) / h;
    auto newWidth = static_cast<UINT>(w * ratio + 0.5);
    return ResizeWICBitmap(bmp, newWidth, newHeight);
}

#define BEGIN_NAMESPACE(x) namespace x {
#define END_NAMESPACE }

BEGIN_NAMESPACE(PPOcr)

TextRecognizer::TextRecognizer(hstring const& model_path, hstring const& dict_path) : TextRecognizer(nullptr) {
    InitDict(dict_path);

    auto model = LearningModel::LoadFromFilePath(model_path);
    auto inFeatures = model.InputFeatures();
    auto outFeatures = model.OutputFeatures();
    if (inFeatures.Size() != 1 || outFeatures.Size() != 1) {
        throw hresult_error(E_FAIL, L"Model must have exactly one input and one output feature.");
    }
    auto inFeatureDesc = inFeatures.GetAt(0);
    auto outFeatureDesc = outFeatures.GetAt(0);
    if (inFeatureDesc.Kind() != LearningModelFeatureKind::Tensor || outFeatureDesc.Kind() != LearningModelFeatureKind::Tensor) {
        throw hresult_error(E_FAIL, L"Model must have a tensor input and a tensor output feature.");
    }
    auto inTensorDesc = inFeatureDesc.as<TensorFeatureDescriptor>();
    auto outTensorDesc = outFeatureDesc.as<TensorFeatureDescriptor>();
    if (inTensorDesc.TensorKind() != TensorKind::Float || outTensorDesc.TensorKind() != TensorKind::Float) {
        throw hresult_error(E_FAIL, L"Model must have a float32 tensor input and a float32 tensor output feature.");
    }
    if (!VectorEquals(inTensorDesc.Shape(), { -1, 3, 48, -1 })) {
        throw hresult_error(E_FAIL, L"Model input tensor shape must be [*, 3, 48, *].");
    }
    if (!VectorEquals(outTensorDesc.Shape(), { -1, -1, 18385 })) {
        throw hresult_error(E_FAIL, L"Model output tensor shape must be [*, *, 18385].");
    }

    // OK
    m_inTensorName = inFeatureDesc.Name();
    m_outTensorName = outFeatureDesc.Name();
    m_model = std::move(model);
}

TextRecognitionFragment TextRecognizer::recognize(SoftwareBitmap const& image) {
    InitSession();

    // Get pointer to the image data
    auto bmp = SoftwareBitmapToWICBitmap(image);
    bmp = ResizeWICBitmapByHeight(bmp, 48);
    auto [bmpWidth, bmpHeight] = GetWICBitmapSize(bmp);
    com_ptr<IWICBitmapLock> bmpLock;
    WICRect rect{ 0, 0, (INT)bmpWidth, (INT)bmpHeight };
    check_hresult(bmp->Lock(&rect, WICBitmapLockRead, bmpLock.put()));
    /*UINT stride;
    check_hresult(bmpLock->GetStride(&stride));*/
    BYTE* bmpData;
    UINT bmpDataSize;
    check_hresult(bmpLock->GetDataPointer(&bmpDataSize, &bmpData));
    if (!bmpData) {
        throw hresult_error(E_FAIL, L"Failed to get bitmap data pointer.");
    }
    assert(bmpDataSize == bmpWidth * bmpHeight * 4); // Assuming BGRA format

    // Prepare the input tensor of shape [batch_size, BGR planes (3), height (48), width]
    m_inTensor = TensorFloat::CreateFromArray(
        { 1, 3, 48, bmpWidth }, // Shape: [1, 3, 48, width]
        {});
    m_outTensor = TensorFloat::Create();
    BYTE* tensorDataRaw;
    UINT tensorCap;
    check_hresult(m_inTensor.as<ITensorNative>()->GetBuffer(&tensorDataRaw, &tensorCap));
    assert(tensorCap >= 1 * 3 * 48 * bmpWidth * sizeof(float));
    auto tensorData = reinterpret_cast<float*>(tensorDataRaw);
    // Convert from B8G8R8A8 to float32[1, 3, 48, width]
    for (uint32_t y = 0; y < 48; y++) {
        for (uint32_t x = 0; x < bmpWidth; x++) {
            // Copy pixel data to the tensor
            auto srcIndex = (y * bmpWidth + x) * 4;
            tensorData[0 * 48 * bmpWidth + y * bmpWidth + x] = bmpData[srcIndex + 0] / 255.0f * 2.0f - 1.0f; // B
            tensorData[1 * 48 * bmpWidth + y * bmpWidth + x] = bmpData[srcIndex + 1] / 255.0f * 2.0f - 1.0f; // G
            tensorData[2 * 48 * bmpWidth + y * bmpWidth + x] = bmpData[srcIndex + 2] / 255.0f * 2.0f - 1.0f; // R
        }
    }

    // Do inference
    m_session.EvaluateFeatures({ { m_inTensorName, m_inTensor }, { m_outTensorName, m_outTensor } }, {});

    // Get the output tensor of shape [batch_size, sequence_length, num_classes]
    auto outShapeObj = m_outTensor.Shape();
    uint32_t outShape[3] = { (uint32_t)outShapeObj.GetAt(0), (uint32_t)outShapeObj.GetAt(1), (uint32_t)outShapeObj.GetAt(2) };
    check_hresult(m_outTensor.as<ITensorNative>()->GetBuffer(&tensorDataRaw, &tensorCap));
    assert(tensorCap >= outShape[0] * outShape[1] * outShape[2] * sizeof(float));

    // Decode the output tensor
    std::vector<std::pair<uint32_t, float>> currentEntries;
    uint32_t prevArgmax{};
    for (uint32_t i = 0; i < outShape[1]; i++) { // For each sequence element
        // Find argmax
        std::span<float> outData(reinterpret_cast<float*>(tensorDataRaw) + i * outShape[2], outShape[2]);
        auto argmax = uint32_t(std::max_element(begin(outData), end(outData)) - begin(outData));
        float confidence = outData[argmax];
        if (argmax == 0) {
            // Skip the blank token
            prevArgmax = argmax;
            continue;
        }

        // Filter duplicate characters
        bool filterDuplicate = true;
        if (filterDuplicate) {
            if (prevArgmax != argmax) {
                currentEntries.emplace_back(argmax, confidence);
            }
        } else {
            currentEntries.emplace_back(argmax, confidence);
        }

        prevArgmax = argmax;
    }
    std::wstring currentText;
    float currentConfidence = 0.0f;
    for (auto& [index, confidence] : currentEntries) {
        currentText += m_ctcDict[index];
        currentConfidence += confidence;
    }

    return {
        .text = hstring(currentText),
        .confidence = currentConfidence / currentEntries.size(),
    };
}

void TextRecognizer::InitSession() {
    if (m_session) {
        // If the session already exists, we can reuse it
        return;
    }

    if (!m_model) {
        throw hresult_error(E_FAIL, L"Model is invalid.");
    }
    if (!m_device) {
        throw hresult_error(E_FAIL, L"Device is invalid.");
    }
    // Create a LearningModelSession with the model and device
    auto session = LearningModelSession(m_model, m_device);
    if (!session) {
        throw hresult_error(E_FAIL, L"Failed to create LearningModelSession.");
    }

    // OK
    m_session = std::move(session);
}

void TextRecognizer::InitDict(hstring const& dictPath) {
    if (dictPath.empty()) {
        // No dictionary provided
        throw hresult_invalid_argument(L"Dictionary path cannot be empty.");
    }

    // Load the dictionary from the file
    std::vector<winrt::hstring> ctcDict;
    ctcDict.reserve(18385); // Assuming the dictionary has 18385 entries
    ctcDict.push_back(L"blank"); // blank token [IGNORED]
    ReadFileByLinesWinAPI(dictPath, [&](std::string&& line) {
        // Convert to hstring and add to the dictionary
        ctcDict.push_back(to_hstring(line));
    });
    ctcDict.push_back(L" "); // The space token
    if (ctcDict.size() != 18385) {
        throw hresult_error(E_FAIL, L"Dictionary must contain exactly 18385 entries.");
    }

    // OK
    m_ctcDict = std::move(ctcDict);
}

END_NAMESPACE

void PPOcr::GlobalInit() {
    g_DxUseCore = true;

    com_ptr<IDXCoreAdapterFactory> factory;
    check_hresult(DXCoreCreateAdapterFactory(guid_of<decltype(factory)>(), factory.put_void()));
    com_ptr<IDXCoreAdapterList> adapterList;
    check_hresult(factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, adapterList.put()));
    auto adapterCount = adapterList->GetAdapterCount();
    if (adapterCount == 0) {
        g_DxGenericMLSupport = false;
    }
    else {
        g_DxGenericMLSupport = true;
    }
}

std::vector<hstring> PPOcr::EnumerateD3D12Devices() {
    if (false) {
        return EnumerateD3D12DevicesViaDxgi();
    }
    else {
        return EnumerateD3D12DevicesViaDxCore();
    }
}

IAsyncOperation<IVector<hstring>> PPOcr::EnumerateD3D12DevicesAsync() {
    co_await resume_background();
    co_return single_threaded_vector(EnumerateD3D12Devices());
}

LearningModelDevice PPOcr::CreateLearningModelDevice(hstring const& device) {
    auto d3dDevice = PickD3D12DeviceForML(device);
    return CreateLearningModelDevice(d3dDevice);
}
