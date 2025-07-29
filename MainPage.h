#pragma once

#include "MainPage.g.h"

#include "PPOcr.hpp"

namespace winrt::PPOcrDemo::implementation {
    struct MainPage : MainPageT<MainPage> {
        MainPage();

        void OcrImageArea_DragOver(Windows::Foundation::IInspectable const& sender, Windows::UI::Xaml::DragEventArgs const& e);
        void OcrImageArea_Drop(Windows::Foundation::IInspectable const& sender, Windows::UI::Xaml::DragEventArgs const& e);
        void OcrImageArea_OnControlV(Windows::UI::Xaml::Input::KeyboardAccelerator const& sender, Windows::UI::Xaml::Input::KeyboardAcceleratorInvokedEventArgs const& e);
        void StartOcrButton_Click(Windows::Foundation::IInspectable const& sender, Windows::UI::Xaml::RoutedEventArgs const& e);
        void ClearOcrOutput_Click(Windows::Foundation::IInspectable const& sender, Windows::UI::Xaml::RoutedEventArgs const& e);

    private:
        Windows::Foundation::IAsyncAction ReinitOcrModelsAsync(bool invalidateDevices);
        hstring GetSelectedInferenceDevice();
        void ReportExceptionAsDialog(hstring const& header = {});
        Windows::Foundation::IAsyncAction LoadInputImageFromDataPackageAsync(Windows::ApplicationModel::DataTransfer::DataPackageView const& dp);
        Windows::Foundation::IAsyncAction DoOcrAsync();

        auto MakeLoadSession();
        template <typename F>
        auto WithLoadSession(F&& f);

        IInspectable m_previousDeviceSelection{ nullptr };
        Windows::Graphics::Imaging::SoftwareBitmap m_inputImage{ nullptr };
        PPOcr::TextRecognizer m_textRecognizer{ nullptr };
    };
}

namespace winrt::PPOcrDemo::factory_implementation {
    struct MainPage : MainPageT<MainPage, implementation::MainPage> {};
}
