#include "pch.h"
#include "MainPage.h"
#include "MainPage.g.cpp"

#include <winrt/Tenkai.UI.Xaml.h>
#include <winrt/Tenkai.UI.ViewManagement.h>

#include <Tenkai.hpp>

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::System;
using namespace Windows::UI::Core;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Input;
using namespace Windows::UI::Xaml::Media;
using namespace Windows::UI::Xaml::Media::Imaging;
using namespace Windows::ApplicationModel::DataTransfer;
using namespace Windows::Storage;
using namespace Windows::Storage::Streams;
using namespace Windows::Graphics::Imaging;

namespace winrt::Tenkai {
    using UI::Xaml::Window;
}

#define DEFAULT_DEVICE_STRING L"自动"

IAsyncOperation<SoftwareBitmap> LoadSoftwareBitmapFromStreamAsync(IRandomAccessStream const& stream) {
    auto decoder = co_await BitmapDecoder::CreateAsync(stream);
    auto softwareBitmap = co_await decoder.GetSoftwareBitmapAsync();
    co_return softwareBitmap;
}

IAsyncOperation<SoftwareBitmap> LoadSoftwareBitmapFromStorageFileAsync(IStorageFile const& file) {
    auto stream = co_await file.OpenAsync(FileAccessMode::Read);
    auto softwareBitmap = co_await LoadSoftwareBitmapFromStreamAsync(stream);
    co_return softwareBitmap;
}

IAsyncOperation<SoftwareBitmap> LoadSoftwareBitmapFromDataPackageAsync(DataPackageView const& dp) {
    if (dp.Contains(StandardDataFormats::Bitmap())) {
        auto bitmap = co_await dp.GetBitmapAsync();
        if (bitmap) {
            auto stream = co_await bitmap.OpenReadAsync();
            auto softwareBitmap = co_await LoadSoftwareBitmapFromStreamAsync(stream);
            co_return softwareBitmap;
        }
    }
    else if (dp.Contains(StandardDataFormats::StorageItems())) {
        auto items = co_await dp.GetStorageItemsAsync();
        for (auto item : items) {
            if (auto file = item.try_as<IStorageFile>()) {
                auto softwareBitmap = co_await LoadSoftwareBitmapFromStorageFileAsync(file);
                co_return softwareBitmap;
            }
        }
    }
    co_return nullptr;
}

namespace winrt::PPOcrDemo::implementation {
    auto MainPage::MakeLoadSession() {
        VisualStateManager::GoToState(*this, L"IndicationLoading", true);
        return tenkai::cpp_utils::ScopeExit([this, dq = DispatcherQueue::GetForCurrentThread()] {
            try {
                if (dq.HasThreadAccess()) {
                    VisualStateManager::GoToState(*this, L"IndicationIdle", true);
                }
                else {
                    dq.TryEnqueue([self = get_strong()] {
                        VisualStateManager::GoToState(*self, L"IndicationIdle", true);
                    });
                }
            }
            catch (...) {}
        });
    }

    template <typename F>
    auto MainPage::WithLoadSession(F&& f) {
        auto loadSess = MakeLoadSession();
        return f();
    }

    MainPage::MainPage() {
        DispatcherQueue::GetForCurrentThread().TryEnqueue(DispatcherQueuePriority::Low, [this] {
            PPOcr::GlobalInit();

            ReinitOcrModelsAsync(true);
        });
    }

    void MainPage::OcrImageArea_DragOver(IInspectable const& sender, DragEventArgs const& e) {
        e.AcceptedOperation(DataPackageOperation::Copy);
    }

    void MainPage::OcrImageArea_Drop(IInspectable const& sender, DragEventArgs const& e) {
        LoadInputImageFromDataPackageAsync(e.DataView());

        auto hwnd = std::bit_cast<HWND>(Tenkai::Window::GetCurrentMain().View().Id());
        SetForegroundWindow(hwnd);
    }

    void MainPage::OcrImageArea_OnControlV(KeyboardAccelerator const& sender, KeyboardAcceleratorInvokedEventArgs const& e) {
        LoadInputImageFromDataPackageAsync(Clipboard::GetContent());
    }

    void MainPage::StartOcrButton_Click(IInspectable const& sender, RoutedEventArgs const& e) {
        DoOcrAsync();
    }

    void MainPage::ClearOcrOutput_Click(IInspectable const& sender, RoutedEventArgs const& e) {
        OcrOutputTextBox().Text({});
        OcrTimingTextBlock().Text({});
    }

    IAsyncAction MainPage::ReinitOcrModelsAsync(bool invalidateDevices) {
        auto self = get_strong();
        auto loadSess = MakeLoadSession();

        try {
            if (invalidateDevices) {
                auto devs = co_await PPOcr::EnumerateD3D12DevicesAsync();

                // Retrieve a list of inference devices
                auto idCb = InferenceDeviceComboBox();
                auto cbItems = idCb.Items();
                cbItems.Clear();
                cbItems.Append(box_value(DEFAULT_DEVICE_STRING));
                cbItems.Append(box_value(L"CPU"));
                for (auto&& dev : devs) {
                    cbItems.Append(box_value(dev));
                }
                idCb.SelectedIndex(0);

                co_return;
            }

            // Load OCR models
            auto devSelection = GetSelectedInferenceDevice();
            co_await resume_background();
            auto dev = PPOcr::CreateLearningModelDevice(devSelection);
            auto textRecognizer = PPOcr::TextRecognizer(L"models/server_rec_infer.onnx", L"models/ppocrv5_dict.txt");
            textRecognizer.set_device(dev);
            m_textRecognizer = std::move(textRecognizer);
        }
        catch (...) {
            ReportExceptionAsDialog(L"无法初始化推理模型");
        }
    }

    hstring MainPage::GetSelectedInferenceDevice() {
        auto idCb = InferenceDeviceComboBox();
        if (idCb.SelectedIndex() < 0) {
            return {};
        }
        hstring result = unbox_value<hstring>(idCb.SelectedItem());
        if (result == DEFAULT_DEVICE_STRING) {
            result.clear();
        }
        return result;
    }

    void MainPage::ReportExceptionAsDialog(hstring const& header) try {
        hstring errMsg;
        try { throw; }
        catch (hresult_error const& ex) {
            errMsg = winrt::format(L"0x{:08X}\n{}", (uint32_t)ex.code(), ex.message());
        }
        catch (std::exception const& ex) {
            errMsg = winrt::to_hstring(ex.what());
        }
        catch (...) {
            errMsg = L"An unknown error occurred.";
        }

        auto run = [header = header, errMsg = std::move(errMsg), self = get_strong()] {
            ContentDialog cd;
            cd.XamlRoot(self->XamlRoot());
            cd.Title(box_value(header));
            cd.Content(box_value(errMsg));
            cd.CloseButtonText(L"确定");
            std::ignore = cd.ShowAsync();
        };
        auto disp = Dispatcher();
        if (disp.HasThreadAccess()) {
            run();
        }
        else {
            disp.RunAsync(CoreDispatcherPriority::Low, std::move(run));
        }
    }
    catch (...) {}

    IAsyncAction MainPage::LoadInputImageFromDataPackageAsync(DataPackageView const& dataPackage) {
        auto self = get_strong();

        auto loadSess = MakeLoadSession();

        try {
            auto bitmap = co_await LoadSoftwareBitmapFromDataPackageAsync(dataPackage);
            if (bitmap) {
                bitmap = SoftwareBitmap::Convert(bitmap, BitmapPixelFormat::Bgra8, BitmapAlphaMode::Premultiplied);
                auto source = SoftwareBitmapSource();
                co_await source.SetBitmapAsync(bitmap);
                OcrInputImage().Source(source);
            }

            m_inputImage = bitmap;
        }
        catch (...) {
            ReportExceptionAsDialog(L"无法加载图像");
        }
    }

    IAsyncAction MainPage::DoOcrAsync() {
        auto self = get_strong();

        {
            // If device selection changed, reload models with new device
            auto selection = InferenceDeviceComboBox().SelectedItem();
            bool needReinit = (selection != m_previousDeviceSelection) ||
                !m_textRecognizer;
            if (needReinit) {
                m_previousDeviceSelection = std::move(selection);
                co_await ReinitOcrModelsAsync(false);
            }
        }

        if (!m_inputImage) {
            // Image not loaded, do nothing
            co_return;
        }

        auto loadSess = MakeLoadSession();

        bool isTextDetectionEnabled = unbox_value_or<bool>(EnableTextDetectionCheckBox().IsChecked(), false);
        bool isTextRecognitionEnabled = unbox_value_or<bool>(EnableTextRecognitionCheckBox().IsChecked(), false);

        try {
            co_await resume_background();
            auto t0 = std::chrono::high_resolution_clock::now();
            auto output = m_textRecognizer.recognize(m_inputImage);
            auto t1 = std::chrono::high_resolution_clock::now();
            co_await Dispatcher();

            OcrTimingTextBlock().Text(std::format(L"推理耗时: {:.3f} ms", std::chrono::duration<double, std::milli>(t1 - t0).count()));

            std::wstring outputText;
            outputText += std::format(L"{:.3f}:\n{}\n", output.confidence, output.text);
            if (!outputText.empty()) {
                outputText.pop_back(); // Remove the last newline character
            }
            OcrOutputTextBox().Text(outputText);
        }
        catch (...) {
            ReportExceptionAsDialog(L"无法执行 OCR");
        }
    }
}
