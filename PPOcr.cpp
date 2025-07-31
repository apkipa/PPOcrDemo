#include "pch.h"
#include "PPOcr.hpp"

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::Graphics;
using namespace Windows::Graphics::Imaging;

using namespace mam;

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

// Source: https://devblogs.microsoft.com/oldnewthing/20230414-00/?p=108051
SoftwareBitmap ToSoftwareBitmap(IWICBitmap* wicBitmap) {
    SoftwareBitmap bitmap{ nullptr };

    auto native = create_instance<
        ISoftwareBitmapNativeFactory>(
            CLSID_SoftwareBitmapNativeFactory);

    check_hresult(native->CreateFromWICBitmap(
        wicBitmap, true, guid_of<SoftwareBitmap>(),
        put_abi(bitmap)));

    return bitmap;
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

namespace {
    using PPOcr::Quad;
    using PPOcr::BinaryMask;

    struct BoundingBox {
        std::pair<uint32_t, uint32_t> topLeft;     // Left-top corner
        std::pair<uint32_t, uint32_t> topRight;    // Right-top corner
        std::pair<uint32_t, uint32_t> bottomRight; // Right-bottom corner
        std::pair<uint32_t, uint32_t> bottomLeft;  // Left-bottom corner
    };

    struct RotatedRect {
        Point center;
        Size size; // Width and height
        float radius;

        float area() const {
            return size.Width * size.Height;
        }
        float length() const {
            return 2 * (size.Width + size.Height);
        }
        RotatedRect extend_size(float extendWidth, float extendHeight) const {
            return RotatedRect{
                center,
                Size{ size.Width + extendWidth, size.Height + extendHeight },
                radius
            };
        }
        RotatedRect extend_frame(float distance) const {
            return extend_size(distance * 2, distance * 2);
        }
    };

    // 形态学操作实现
    namespace Morphology {
        // 膨胀
        BinaryMask dilate(BinaryMask const& input, int kernelSize) {
            BinaryMask output(input.getWidth(), input.getHeight());
            int halfKernel = kernelSize / 2;

            for (int y = 0; y < (int)input.getHeight(); y++) {
                for (int x = 0; x < (int)input.getWidth(); x++) {
                    if (input.get(x, y)) {
                        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                                if (input.isValid(x + kx, y + ky)) {
                                    output.set(x + kx, y + ky, true);
                                }
                            }
                        }
                    }
                }
            }
            return output;
        }

        // 腐蚀
        BinaryMask erode(BinaryMask const& input, int kernelSize) {
            BinaryMask output(input.getWidth(), input.getHeight());
            int halfKernel = kernelSize / 2;

            for (int y = 0; y < (int)input.getHeight(); ++y) {
                for (int x = 0; x < (int)input.getWidth(); ++x) {
                    bool allWhite = true;
                    for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
                        for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                            if (input.isValid(x + kx, y + ky)) {
                                if (input.get(x + kx, y + ky) == 0) {
                                    allWhite = false;
                                    break;
                                }
                            }
                            else { // 边界外的像素视为黑色
                                allWhite = false;
                                break;
                            }
                        }
                        if (!allWhite) { break; }
                    }
                    if (allWhite) {
                        output.set(x, y, 255);
                    }
                }
            }
            return output;
        }

        // 闭运算：填充小的空洞和断裂
        BinaryMask close(const BinaryMask& input, int kernelSize) {
            BinaryMask dilated = dilate(input, kernelSize);
            return erode(dilated, kernelSize);
        }
    }

    // 并查集 (用于CCL)
    class DSU {
        std::vector<int> parent;
    public:
        DSU(int n) {
            parent.resize(n);
            for (int i = 0; i < n; ++i) parent[i] = i;
        }
        int find(int i) {
            if (parent[i] == i) { return i; }
            return parent[i] = find(parent[i]);
        }
        void unite(int i, int j) {
            int root_i = find(i);
            int root_j = find(j);
            if (root_i != root_j) {
                parent[root_i] = root_j;
            }
        }
    };

    // CCL 实现 (BFS 版本)
    std::vector<std::vector<PointInt32>> findConnectedComponentsBFS(BinaryMask const& mask, uint32_t minArea) {
        int width = mask.getWidth();
        int height = mask.getHeight();
        std::vector<uint8_t> visited(width * height, 0);
        std::vector<std::vector<PointInt32>> components;

        const int dx[4] = { 1, -1, 0, 0 };
        const int dy[4] = { 0, 0, 1, -1 };

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask.get(x, y) && !visited[y * width + x]) {
                    std::vector<PointInt32> component;
                    std::queue<PointInt32> q;
                    q.push({ x, y });
                    visited[y * width + x] = 1;

                    while (!q.empty()) {
                        auto p = q.front();
                        q.pop();
                        component.push_back(p);
                        for (size_t d = 0; d < std::size(dx); d++) {
                            int nx = p.X + dx[d];
                            int ny = p.Y + dy[d];
                            if (mask.isValid(nx, ny) && mask.get(nx, ny) && !visited[ny * width + nx]) {
                                visited[ny * width + nx] = 1;
                                q.push({ nx, ny });
                            }
                        }
                    }
                    if (component.size() >= minArea) {
                        components.push_back(std::move(component));
                    }
                }
            }
        }
        return components;
    }

#if 0
    // CCL 实现
    std::vector<std::vector<PointInt32>> findConnectedComponentsDSU(BinaryMask const& mask, int minArea) {
        int width = mask.getWidth();
        int height = mask.getHeight();
        BinaryMask labeledMask(width, height);
        DSU dsu(width * height);
        int nextLabel = 1;

        // 第一遍：分配初始标签并记录等价关系
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask.get(x, y) == 255) {
                    std::vector<int> neighbors;
                    if (y > 0 && labeledMask.get(x, y - 1) > 0) { neighbors.push_back(labeledMask.get(x, y - 1)); }
                    if (x > 0 && labeledMask.get(x - 1, y) > 0) { neighbors.push_back(labeledMask.get(x - 1, y)); }

                    if (neighbors.empty()) {
                        labeledMask.set(x, y, nextLabel);
                        nextLabel++;
                    }
                    else {
                        int minNeighbor = neighbors[0];
                        for (size_t i = 1; i < neighbors.size(); i++) {
                            if (neighbors[i] < minNeighbor) { minNeighbor = neighbors[i]; }
                        }
                        labeledMask.set(x, y, minNeighbor);
                        for (int neighbor : neighbors) {
                            dsu.unite(minNeighbor, neighbor);
                        }
                    }
                }
            }
        }

        // 第二遍：解析等价关系并收集点
        std::map<int, std::vector<PointInt32>> components;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int label = labeledMask.get(x, y);
                if (label > 0) {
                    int rootLabel = dsu.find(label);
                    components[rootLabel].push_back({ x, y });
                }
            }
        }

        // 过滤小面积组件
        std::vector<std::vector<PointInt32>> filteredComponents;
        for (auto&& [label, points] : components) {
            if (points.size() >= minArea) {
                filteredComponents.push_back(std::move(points));
            }
        }

        return filteredComponents;
    }
#endif

    namespace Geometry {
        // 向量叉积，用于判断转向
        long long cross_product(PointInt32 O, PointInt32 A, PointInt32 B) {
            return (long long)(A.X - O.X) * (B.Y - O.Y) - (long long)(A.Y - O.Y) * (B.X - O.X);
        }

        // 计算凸包 (Monotone Chain Algorithm)
        std::vector<PointInt32> convexHull(std::vector<PointInt32>& points) {
            size_t n = points.size();
            if (n <= 3) { return points; }

            std::sort(points.begin(), points.end(), [](PointInt32 A, PointInt32 B) {
                return A.X < B.X || (A.X == B.X && A.Y < B.Y);
            });

            std::vector<PointInt32> hull;
            // 构建下凸包
            for (size_t i = 0; i < n; i++) {
                while (hull.size() >= 2 && cross_product(hull[hull.size() - 2], hull.back(), points[i]) <= 0) {
                    hull.pop_back();
                }
                hull.push_back(points[i]);
            }

            // 构建上凸包
            for (size_t i = n - 2, t = hull.size() + 1; i != -1; i--) {
                while (hull.size() >= t && cross_product(hull[hull.size() - 2], hull.back(), points[i]) <= 0) {
                    hull.pop_back();
                }
                hull.push_back(points[i]);
            }

            hull.pop_back(); // 最后一个点是重复的
            return hull;
        }

        // 计算两点间距离的平方
        double distSq(Point p1, Point p2) {
            return (p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y);
        }

        // 旋转卡尺法寻找最小外接矩形，返回最小外接矩形的旋转矩形参数
        RotatedRect getMinimumBoundingRectangle(std::vector<PointInt32> const& hull) {
            double minArea = std::numeric_limits<double>::max();
            RotatedRect bestRect{};
            size_t n = hull.size();
            if (n < 3) { return {}; } // 无法形成矩形

            for (size_t i = 0; i < n; i++) {
                PointInt32 p1 = hull[i];
                PointInt32 p2 = hull[(i + 1) % n];

                // 边的方向向量
                double edge_dx = p2.X - p1.X;
                double edge_dy = p2.Y - p1.Y;
                double edge_len = std::sqrt(edge_dx * edge_dx + edge_dy * edge_dy);
                if (edge_len == 0) { continue; }

                // 单位方向向量
                double ux = edge_dx / edge_len;
                double uy = edge_dy / edge_len;
                // 单位法向量
                double vx = -uy;
                double vy = ux;

                double min_proj_edge = 0, max_proj_edge = 0;
                double min_proj_norm = 0, max_proj_norm = 0;

                for (auto const& p : hull) {
                    double proj_edge = (p.X - p1.X) * ux + (p.Y - p1.Y) * uy;
                    double proj_norm = (p.X - p1.X) * vx + (p.Y - p1.Y) * vy;

                    if (proj_edge < min_proj_edge) { min_proj_edge = proj_edge; }
                    if (proj_edge > max_proj_edge) { max_proj_edge = proj_edge; }
                    if (proj_norm < min_proj_norm) { min_proj_norm = proj_norm; }
                    if (proj_norm > max_proj_norm) { max_proj_norm = proj_norm; }
                }

                double width = max_proj_edge - min_proj_edge;
                double height = max_proj_norm - min_proj_norm;
                double area = width * height;

                if (area < minArea) {
                    minArea = area;
                    // 计算中心点
                    double cx = p1.X + (min_proj_edge + width / 2) * ux + (min_proj_norm + height / 2) * vx;
                    double cy = p1.Y + (min_proj_edge + width / 2) * uy + (min_proj_norm + height / 2) * vy;
                    // 旋转角度（以x轴为基准，逆时针，弧度）
                    float angle = static_cast<float>(std::atan2(uy, ux));
                    bestRect.center = { static_cast<float>(cx), static_cast<float>(cy) };
                    bestRect.size = { static_cast<float>(width), static_cast<float>(height) };
                    bestRect.radius = angle;
                }
            }
            return bestRect;
        }
    }

    Quad ToQuad(RotatedRect const& rect) {
        // Calculate half dimensions
        float halfWidth = rect.size.Width * 0.5f;
        float halfHeight = rect.size.Height * 0.5f;
        // Precompute sin/cos
        float cosTheta = std::cos(rect.radius);
        float sinTheta = std::sin(rect.radius);
        // Define unrotated corners relative to center
        constexpr int dx[4] = { -1,  1,  1, -1 };
        constexpr int dy[4] = { -1, -1,  1,  1 };
        Point corners[4];
        for (int i = 0; i < 4; ++i) {
            float x = dx[i] * halfWidth;
            float y = dy[i] * halfHeight;
            // Rotate and translate
            corners[i].X = rect.center.X + x * cosTheta - y * sinTheta;
            corners[i].Y = rect.center.Y + x * sinTheta + y * cosTheta;
        }
        return Quad{ { corners[0], corners[1], corners[2], corners[3] } };
    }
}

/// <summary>
/// 查找图像中的轮廓区域，返回每个区域的外接包围盒。
/// <typeparam name="F">fn(RotatedRect)</typeparam>
/// <param name="mask"></param>
/// <param name="f">回调，每个区域调用一次</param>
template <typename F>
void FindBoundingBoxes(BinaryMask const& mask, F&& f) {
    // Step 1: Find connected components (regions) in the mask
    constexpr uint32_t kMinArea = 10; // Minimum area to filter noise, can be parameterized
    auto components = findConnectedComponentsBFS(mask, kMinArea);

    // Step 2: For each component, compute convex hull and minimum bounding rectangle
    for (auto& comp : components) {
        if (comp.empty()) { continue; }
        // Compute convex hull
        auto hull = Geometry::convexHull(comp);
        if (hull.size() < 3) { continue; } // Ignore degenerate
        // Compute minimum bounding rectangle
        auto minRect = Geometry::getMinimumBoundingRectangle(hull);
        f(std::move(minRect));
    }
}

#define BEGIN_NAMESPACE(x) namespace x {
#define END_NAMESPACE }

BEGIN_NAMESPACE(PPOcr)

TextDetector::TextDetector(hstring const& model_path) : m_model(nullptr), m_device(nullptr) {
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
    if (!VectorEquals(inTensorDesc.Shape(), { -1, 3, -1, -1 })) {
        throw hresult_error(E_FAIL, L"Model input tensor shape must be [*, 3, *, *].");
    }
    if (auto shape = outTensorDesc.Shape(); !VectorEquals(shape, { -1, 1, -1, -1 }) && !VectorEquals(shape, { -1, -1, -1, -1 })) {
        throw hresult_error(E_FAIL, L"Model output tensor shape must be [*, 1, *, *].");
    }

    // OK
    m_inTensorName = inFeatureDesc.Name();
    m_outTensorName = outFeatureDesc.Name();
    m_model = std::move(model);
}

std::vector<Quad> TextDetector::detect(SoftwareBitmap const& image) {
    InitSession();

    auto imgMask = DetectMaskImage(image);

    std::vector<Quad> results;
    FindBoundingBoxes(imgMask, [&](RotatedRect&& box) {
        // Use the area and length to calculate the distance for extending the bounding box
        const float unclipRatio = 2.0f;
        float distance = box.area() * unclipRatio / box.length();
        auto quad = ToQuad(box.extend_frame(distance));
        //auto quad = ToQuad(box);
        results.push_back(quad);
    });

    return results;
}

SoftwareBitmap TextDetector::detect_mask(SoftwareBitmap const& image) {
    InitSession();

    auto imgWidth = (uint32_t)image.PixelWidth();
    auto imgHeight = (uint32_t)image.PixelHeight();
    auto imgMask = DetectMaskImage(image);
    auto maskWidth = imgMask.getWidth();
    auto maskHeight = imgMask.getHeight();

    SoftwareBitmap outImage(BitmapPixelFormat::Gray8, imgWidth, imgHeight);
    auto buffer = outImage.LockBuffer(BitmapBufferAccessMode::Write);
    auto planeDescription = buffer.GetPlaneDescription(0);
    auto stride = planeDescription.Stride;
    auto reference = buffer.CreateReference();
    auto ptr = reference.data();
    for (uint32_t y = 0; y < std::min(imgHeight, maskHeight); y++) {
        for (uint32_t x = 0; x < std::min(imgWidth, maskWidth); x++) {
            // Set the pixel value based on the mask
            uint8_t value = imgMask.get(x, y);
            ptr[y * stride + x] = value;
        }
    }

    return outImage;
}

BinaryMask TextDetector::DetectMaskImage(SoftwareBitmap const& image) {
    // Get pointer to the image data
    auto bmp = SoftwareBitmapToWICBitmap(image);
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

    // Prepare the input tensor of shape [batch_size, channels (3), height, width]
    // Resize the image to a multiple of 32 pixels in height and width as required by the model
    auto inHeight = (bmpHeight + 31) / 32 * 32;
    auto inWidth = (bmpWidth + 31) / 32 * 32;
    inHeight = std::max(inHeight, 32u); // Ensure height is at least 32
    inWidth = std::max(inWidth, 32u); // Ensure width is at least 32
    m_inTensor = TensorFloat::CreateFromArray(
        { 1, 3, inHeight, inWidth }, // Shape: [1, 3, height, width]
        {});
    m_outTensor = TensorFloat::Create();
    BYTE* tensorDataRaw;
    UINT tensorCap;
    check_hresult(m_inTensor.as<ITensorNative>()->GetBuffer(&tensorDataRaw, &tensorCap));
    assert(tensorCap >= 1 * 3 * inHeight * inWidth * sizeof(float));
    auto tensorData = reinterpret_cast<float*>(tensorDataRaw);
    // Convert from B8G8R8A8 to float32[1, 3, height, width]
    for (uint32_t y = 0; y < std::min(bmpHeight, inHeight); y++) {
        for (uint32_t x = 0; x < std::min(bmpWidth, inWidth); x++) {
            // Normalize pixel values to ImageNet range
            // NOTE: Here are proper BGR-ordered params, see https://github.com/PaddlePaddle/PaddleOCR/issues/6070
            float scale = 1.0f / 255.0f; // Scale to [0, 1]
            float means[] = { 0.485f, 0.456f, 0.406f }; // ImageNet means for BGR
            float stds[] = { 0.229f, 0.224f, 0.225f }; // ImageNet stds for BGR
            auto srcIndex = (y * bmpWidth + x) * 4;
            tensorData[0 * inHeight * inWidth + y * inWidth + x] = (bmpData[srcIndex + 0] * scale - means[0]) / stds[0]; // B
            tensorData[1 * inHeight * inWidth + y * inWidth + x] = (bmpData[srcIndex + 1] * scale - means[1]) / stds[1]; // G
            tensorData[2 * inHeight * inWidth + y * inWidth + x] = (bmpData[srcIndex + 2] * scale - means[2]) / stds[2]; // R
        }
    }

    // Do inference
    m_session.EvaluateFeatures({ { m_inTensorName, m_inTensor }, { m_outTensorName, m_outTensor } }, {});

    // Get the output tensor of shape [batch_size, channels, height, width]
    auto outShapeObj = m_outTensor.Shape();
    uint32_t outShape[4] = { (uint32_t)outShapeObj.GetAt(0), (uint32_t)outShapeObj.GetAt(1), (uint32_t)outShapeObj.GetAt(2), (uint32_t)outShapeObj.GetAt(3) };
    check_hresult(m_outTensor.as<ITensorNative>()->GetBuffer(&tensorDataRaw, &tensorCap));
    tensorData = reinterpret_cast<float*>(tensorDataRaw);
    assert(tensorCap >= outShape[0] * outShape[1] * outShape[2] * outShape[3] * sizeof(float));
    assert(outShape[0] == 1); // Batch size must be 1
    assert(outShape[1] == 1); // Channels must be 1 (binary mask)

    const float threshold = 0.7f;

    auto outHeight = outShape[2];
    auto outWidth = outShape[3];
    std::vector<uint8_t> maskData(outHeight * outWidth, 0);
    for (uint32_t y = 0; y < outHeight; y++) {
        for (uint32_t x = 0; x < outWidth; x++) {
            // Check if the pixel is above the threshold
            if (tensorData[y * outWidth + x] > threshold) {
                maskData[y * outWidth + x] = 255; // Set to white
            }
            else {
                maskData[y * outWidth + x] = 0; // Set to black
            }
        }
    }

    return { outWidth, outHeight, std::move(maskData) }; // Return mask data and dimensions
}

void TextDetector::InitSession() {
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

    // OK
    m_session = std::move(session);
}

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
        }
        else {
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
    if (g_DxUseCore) {
        return EnumerateD3D12DevicesViaDxCore();
    }
    else {
        return EnumerateD3D12DevicesViaDxgi();
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
