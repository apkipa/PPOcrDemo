#pragma once

#include "pch.h"

#ifdef USE_WINML_NUGET
namespace mam = winrt::Microsoft::AI::MachineLearning;
#else
namespace mam = winrt::Windows::AI::MachineLearning;
#endif

namespace PPOcr {
    struct TextRecognitionFragment {
        winrt::hstring text;
        float confidence;
    };

    struct TextRecognitionFragmentWithBox : TextRecognitionFragment {
        winrt::Windows::Foundation::Rect bounding_box;
    };

    struct TextRecognitionOutput {
        std::vector<TextRecognitionFragmentWithBox> entries;
    };

    struct TextDetector {
        TextDetector(std::nullptr_t) : m_model(nullptr), m_device(nullptr) {}
        TextDetector(winrt::hstring const& model_path);

        operator bool() const { return (bool)m_model; }

        void set_device(mam::LearningModelDevice const& device) {
            m_device = device;
            m_session = nullptr; // Reset session to reinitialize with new device
        }

        std::vector<winrt::Windows::Foundation::Rect> detect(winrt::Windows::Graphics::Imaging::SoftwareBitmap const& image);
        winrt::Windows::Graphics::Imaging::SoftwareBitmap detect_mask(winrt::Windows::Graphics::Imaging::SoftwareBitmap const& image);

    private:
        void InitSession();
        std::tuple<std::vector<uint8_t>, uint32_t, uint32_t> DetectMaskImage(winrt::Windows::Graphics::Imaging::SoftwareBitmap const& image);

        mam::LearningModel m_model{ nullptr };
        mam::LearningModelDevice m_device{ nullptr };
        mam::LearningModelSession m_session{ nullptr };
        winrt::hstring m_inTensorName, m_outTensorName;
        mam::TensorFloat m_inTensor{ nullptr }, m_outTensor{ nullptr };
    };

    struct TextRecognizer {
        TextRecognizer(std::nullptr_t) : m_model(nullptr), m_device(nullptr) {}
        TextRecognizer(winrt::hstring const& model_path, winrt::hstring const& dict_path);

        operator bool() const { return (bool)m_model; }

        void set_device(mam::LearningModelDevice const& device) {
            m_device = device;
            m_session = nullptr; // Reset session to reinitialize with new device
        }

        TextRecognitionFragment recognize(winrt::Windows::Graphics::Imaging::SoftwareBitmap const& image);
        std::vector<TextRecognitionFragment> recognize_many(winrt::array_view<winrt::Windows::Graphics::Imaging::SoftwareBitmap const> images);

    private:
        void InitSession();
        void InitDict(winrt::hstring const& dictPath);

        std::vector<winrt::hstring> m_ctcDict;
        mam::LearningModel m_model{ nullptr };
        mam::LearningModelDevice m_device{ nullptr };
        mam::LearningModelSession m_session{ nullptr };
        winrt::hstring m_inTensorName, m_outTensorName;
        mam::TensorFloat m_inTensor{ nullptr }, m_outTensor{ nullptr };
    };

    struct PPOcr {
        PPOcr() : m_detector(nullptr), m_recognizer(nullptr) {}

        void set_device(mam::LearningModelDevice const& device) {
            m_detector.set_device(device);
            m_recognizer.set_device(device);
        }

        void set_text_detector(TextDetector const& detector);
        void set_text_recognizer(TextRecognizer const& recognizer);

        TextRecognitionOutput do_ocr(winrt::Windows::Graphics::Imaging::SoftwareBitmap const& image);

    private:
        TextDetector m_detector;
        TextRecognizer m_recognizer;
    };

    void GlobalInit();
    std::vector<winrt::hstring> EnumerateD3D12Devices();
    winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Foundation::Collections::IVector<winrt::hstring>> EnumerateD3D12DevicesAsync();
    mam::LearningModelDevice CreateLearningModelDevice(winrt::hstring const& device);

}
