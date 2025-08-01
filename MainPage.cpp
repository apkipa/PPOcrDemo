#include "pch.h"
#include "MainPage.h"
#include "MainPage.g.cpp"

#include <winrt/Tenkai.UI.Xaml.h>
#include <winrt/Tenkai.UI.ViewManagement.h>

#include <Tenkai.hpp>

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::System;
using namespace Windows::UI;
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

    void MainPage::OcrDetectionLayoutRoot_RightTapped(IInspectable const& sender, RightTappedRoutedEventArgs const& e) {
        auto originalSource = e.OriginalSource().try_as<Shapes::Polygon>();
        if (!originalSource) {
            return;
        }

        e.Handled(true);

        // Extract the polygon data
        auto points = originalSource.Points();
        PPOcr::Quad quad{};
        for (uint32_t i = 0; i < 4 && i < points.Size(); ++i) {
            quad.corners[i] = points.GetAt(i);
        }

        if (false) {
            auto quadWidth = std::hypot(quad.corners[1].X - quad.corners[0].X, quad.corners[1].Y - quad.corners[0].Y);
            auto quadHeight = std::hypot(quad.corners[3].X - quad.corners[0].X, quad.corners[3].Y - quad.corners[0].Y);
            if (quadHeight / quadWidth >= 1.2) {
                // Handle vertical text
                std::rotate(std::begin(quad.corners), std::begin(quad.corners) + 1, std::end(quad.corners));
            }
        }

        // Extract the image from quad
        auto imgQuad = PPOcr::ExtractQuadFromBitmap(m_inputImage, quad);
        auto imgSrc = SoftwareBitmapSource();
        std::ignore = imgSrc.SetBitmapAsync(imgQuad);

        // Show context menu
        Flyout menu;
        Image imgCtrl;
        imgCtrl.Source(imgSrc);
        menu.Content(imgCtrl);
        Primitives::FlyoutShowOptions menuShowOptions;
        auto cursorPos = e.GetPosition(originalSource);
        cursorPos.Y -= 16.0f;
        menuShowOptions.Position(cursorPos);
        menu.ShowAt(originalSource, menuShowOptions);
    }

    void MainPage::StartOcrButton_Click(IInspectable const& sender, RoutedEventArgs const& e) {
        DoOcrAsync(false);
    }

    void MainPage::ShowOcrMaskButton_Click(IInspectable const& sender, RoutedEventArgs const& e) {
        DoOcrAsync(true);
    }

    void MainPage::ClearOcrOutput_Click(IInspectable const& sender, RoutedEventArgs const& e) {
        OcrInputMaskImage().Source(nullptr);
        MarkDetectionQuads({});

        OcrOutputTextBox().Text({});
        OcrTimingTextBlock().Text({});
    }

    void MainPage::GridSplitter_DoubleTapped(IInspectable const& sender, DoubleTappedRoutedEventArgs const& e) {
        e.Handled(true);

        // Restore width
        LayoutRoot().ColumnDefinitions().GetAt(1).Width(GridLengthHelper::FromPixels(400));
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
            auto textDetector = PPOcr::TextDetector(L"models/server_det_infer.onnx");
            auto textRecognizer = PPOcr::TextRecognizer(L"models/server_rec_infer.onnx", L"models/ppocrv5_dict.txt");
            textDetector.set_device(dev);
            textRecognizer.set_device(dev);
            m_textDetector = std::move(textDetector);
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
            try {
                std::ignore = cd.ShowAsync();
            }
            catch (...) {}
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

    IAsyncAction MainPage::DoOcrAsync(bool showMask) {
        auto self = get_strong();

        {
            // If device selection changed, reload models with new device
            auto selection = InferenceDeviceComboBox().SelectedItem();
            bool needReinit = (selection != m_previousDeviceSelection) ||
                !m_textDetector || !m_textRecognizer;
            if (needReinit) {
                m_previousDeviceSelection = std::move(selection);
                co_await ReinitOcrModelsAsync(false);

                //OcrInputMaskImage().Source(nullptr);
            }
        }

        if (!m_inputImage) {
            // Image not loaded, do nothing
            co_return;
        }

        bool isTextDetectionEnabled = unbox_value_or<bool>(EnableTextDetectionCheckBox().IsChecked(), false);
        bool isTextRecognitionEnabled = unbox_value_or<bool>(EnableTextRecognitionCheckBox().IsChecked(), false);
        bool optimizeModelIO = unbox_value_or<bool>(OptimizeModelIOCheckBox().IsChecked(), false);

        if (!isTextRecognitionEnabled && !isTextDetectionEnabled) {
            // Neither text detection nor recognition is enabled, do nothing
            co_return;
        }

        auto loadSess = MakeLoadSession();

        m_textDetector.set_optimize_resource(optimizeModelIO);
        m_textRecognizer.set_optimize_resource(optimizeModelIO);

        try {
            auto dq = DispatcherQueue::GetForCurrentThread();

            co_await resume_background();
            auto t0 = std::chrono::high_resolution_clock::now();
            if (showMask) {
                auto output = m_textDetector.detect_mask(m_inputImage);
                auto t1 = std::chrono::high_resolution_clock::now();
                auto timing = std::chrono::duration<double, std::milli>(t1 - t0).count();
                co_await resume_foreground(dq);

                output = SoftwareBitmap::Convert(output, BitmapPixelFormat::Bgra8, BitmapAlphaMode::Ignore);
                auto source = SoftwareBitmapSource();
                co_await source.SetBitmapAsync(output);

                OcrTimingTextBlock().Text(winrt::format(L"推理耗时: {:.3f} ms", timing));
                OcrInputMaskImage().Source(source);
            }
            else if (isTextDetectionEnabled && isTextRecognitionEnabled) {
                // Both text detection and recognition enabled
                PPOcr::TextRecognitionOutput output;
                if (optimizeModelIO) {
                    auto ocr = PPOcr::PPOcr();
                    ocr.set_text_detector(m_textDetector);
                    ocr.set_text_recognizer(m_textRecognizer);
                    output = ocr.do_ocr(m_inputImage);
                }
                else {
                    // Handle manually
                    auto textQuads = m_textDetector.detect(m_inputImage);
                    size_t totalProgress = size(textQuads);
                    size_t currentProgress{};
                    auto updateProgressFn = [&] {
                        dq.TryEnqueue([this, msg = winrt::format(L"进度: {} / {}", ++currentProgress, totalProgress)] {
                            OcrTimingTextBlock().Text(msg);
                        });
                    };
                    dq.TryEnqueue([this, textQuads] {
                        MarkDetectionQuads(textQuads);
                    });
                    updateProgressFn();
                    for (auto& quad : textQuads) {
                        auto quadWidth = std::hypot(quad.corners[1].X - quad.corners[0].X, quad.corners[1].Y - quad.corners[0].Y);
                        auto quadHeight = std::hypot(quad.corners[3].X - quad.corners[0].X, quad.corners[3].Y - quad.corners[0].Y);
                        if (quadHeight / quadWidth >= 1.2) {
                            // Handle vertical text
                            std::rotate(std::begin(quad.corners), std::begin(quad.corners) + 1, std::end(quad.corners));
                        }
                        auto img = PPOcr::ExtractQuadFromBitmap(m_inputImage, quad);
                        auto singleOutput = m_textRecognizer.recognize(img);
                        output.entries.push_back({ std::move(singleOutput), quad });
                        updateProgressFn();
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                auto timing = std::chrono::duration<double, std::milli>(t1 - t0).count();
                co_await resume_foreground(dq);

                OcrTimingTextBlock().Text(winrt::format(L"推理耗时: {:.3f} ms", timing));

                std::wstring outputText;
                for (auto& entry : output.entries) {
                    auto& quad = entry.bounding_box;
                    outputText += std::format(
                        L"{:.3f} [({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f})]:\n{}\n",
                        entry.confidence,
                        quad.corners[0].X, quad.corners[0].Y,
                        quad.corners[1].X, quad.corners[1].Y,
                        quad.corners[2].X, quad.corners[2].Y,
                        quad.corners[3].X, quad.corners[3].Y,
                        entry.text
                    );
                }
                if (!outputText.empty()) {
                    outputText.pop_back(); // Remove the last newline character
                }
                OcrOutputTextBox().Text(outputText);
            }
            else if (isTextDetectionEnabled && !isTextRecognitionEnabled) {
                // Only text detection enabled
                auto output = m_textDetector.detect(m_inputImage);
                auto t1 = std::chrono::high_resolution_clock::now();
                auto timing = std::chrono::duration<double, std::milli>(t1 - t0).count();

                dq.TryEnqueue([this, self = std::move(self), timing, output = std::move(output)] {
                    OcrTimingTextBlock().Text(winrt::format(L"推理耗时: {:.3f} ms", timing));

                    std::wstring outputText;
                    for (auto quad : output) {
                        outputText += std::format(L"[({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f})]\n",
                            quad.corners[0].X, quad.corners[0].Y,
                            quad.corners[1].X, quad.corners[1].Y,
                            quad.corners[2].X, quad.corners[2].Y,
                            quad.corners[3].X, quad.corners[3].Y
                        );
                    }
                    if (!outputText.empty()) {
                        outputText.pop_back(); // Remove the last newline character
                    }
                    OcrOutputTextBox().Text(outputText);

                    MarkDetectionQuads(output);
                });
            }
            else if (!isTextDetectionEnabled && isTextRecognitionEnabled) {
                // Only text recognition enabled
                auto output = m_textRecognizer.recognize(m_inputImage);
                auto t1 = std::chrono::high_resolution_clock::now();
                auto timing = std::chrono::duration<double, std::milli>(t1 - t0).count();

                dq.TryEnqueue([this, self = std::move(self), timing, output = std::move(output)] {
                    OcrTimingTextBlock().Text(winrt::format(L"推理耗时: {:.3f} ms", timing));

                    std::wstring outputText;
                    outputText += std::format(L"{:.3f}:\n{}\n", output.confidence, output.text);
                    if (!outputText.empty()) {
                        outputText.pop_back(); // Remove the last newline character
                    }
                    OcrOutputTextBox().Text(outputText);
                });
            }
        }
        catch (...) {
            ReportExceptionAsDialog(L"无法执行 OCR");
        }
    }

    void MainPage::MarkDetectionQuads(std::vector<PPOcr::Quad> const& quads) {
        auto layoutRoot = OcrDetectionLayoutRoot();
        auto layoutRootChildren = layoutRoot.Children();
        layoutRootChildren.Clear();

        if (!m_inputImage) { return; }

        auto canvasWidth = m_inputImage.PixelWidth();
        auto canvasHeight = m_inputImage.PixelHeight();

        layoutRoot.Width(canvasWidth);
        layoutRoot.Height(canvasHeight);

        auto polyBrush = SolidColorBrush();
        polyBrush.Color(Colors::MediumPurple());
        auto edgeBrush = SolidColorBrush();
        edgeBrush.Color(Colors::Red());
        for (auto& quad : quads) {
            Shapes::Polygon polygon;
            polygon.Fill(polyBrush);
            polygon.Points().ReplaceAll(quad.corners);
            layoutRootChildren.Append(polygon);
            Shapes::Line leftEdge;
            leftEdge.Stroke(edgeBrush);
            leftEdge.StrokeThickness(2.0);
            leftEdge.X1(quad.corners[0].X);
            leftEdge.Y1(quad.corners[0].Y);
            leftEdge.X2(quad.corners[1].X);
            leftEdge.Y2(quad.corners[1].Y);
            layoutRootChildren.Append(leftEdge);
        }
    }
}
