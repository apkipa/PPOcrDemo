#pragma once

#include "pch.h"

namespace PPOcr {
    struct TextRecognitionFragment {
        winrt::hstring text;
        float confidence;
    };

    struct TextRecognitionOutput {
        std::vector<TextRecognitionFragment> entries;
    };

    struct TextDetector {
        TextDetector(std::nullptr_t) : m_model(nullptr), m_device(nullptr) {}
        TextDetector(winrt::hstring const& model_path);

        operator bool() const { return (bool)m_model; }

        void set_device(winrt::Windows::AI::MachineLearning::LearningModelDevice const& device) {
            m_device = device;
        }

    private:
        winrt::Windows::AI::MachineLearning::LearningModel m_model{ nullptr };
        winrt::Windows::AI::MachineLearning::LearningModelDevice m_device{ nullptr };
        winrt::Windows::AI::MachineLearning::LearningModelSession m_session{ nullptr };
    };

    struct TextRecognizer {
        TextRecognizer(std::nullptr_t) : m_model(nullptr), m_device(nullptr) {}
        TextRecognizer(winrt::hstring const& model_path, winrt::hstring const& dict_path);

        operator bool() const { return (bool)m_model; }

        void set_device(winrt::Windows::AI::MachineLearning::LearningModelDevice const& device) {
            m_device = device;
            m_session = nullptr; // Reset session to reinitialize with new device
        }

        TextRecognitionOutput recognize(winrt::Windows::Graphics::Imaging::SoftwareBitmap const& image);
        std::vector<TextRecognitionOutput> recognize_many(winrt::array_view<winrt::Windows::Graphics::Imaging::SoftwareBitmap const> images);

    private:
        void InitSession();
        void InitDict(winrt::hstring const& dictPath);

        std::vector<winrt::hstring> m_ctcDict;
        winrt::Windows::AI::MachineLearning::LearningModel m_model{ nullptr };
        winrt::Windows::AI::MachineLearning::LearningModelDevice m_device{ nullptr };
        winrt::Windows::AI::MachineLearning::LearningModelSession m_session{ nullptr };
        winrt::hstring m_inTensorName, m_outTensorName;
        winrt::Windows::AI::MachineLearning::TensorFloat m_inTensor{ nullptr }, m_outTensor{ nullptr };
    };

    struct PPOcr {
        PPOcr() : m_detector(nullptr), m_recognizer(nullptr) {}

        void set_device(winrt::Windows::AI::MachineLearning::LearningModelDevice const& device) {
            m_detector.set_device(device);
            m_recognizer.set_device(device);
        }

        void set_text_detector(TextDetector const& detector);
        void set_text_recognizer(TextRecognizer const& recognizer);

        TextRecognitionOutput recognize(winrt::Windows::Graphics::Imaging::SoftwareBitmap const& image);

    private:
        TextDetector m_detector;
        TextRecognizer m_recognizer;
    };

    void GlobalInit();
    std::vector<winrt::hstring> EnumerateD3D12Devices();
    winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Foundation::Collections::IVector<winrt::hstring>> EnumerateD3D12DevicesAsync();
    winrt::Windows::AI::MachineLearning::LearningModelDevice CreateLearningModelDevice(winrt::hstring const& device);

}
