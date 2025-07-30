# PPOcrDemo

轻量级 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 演示程序，基于 Windows 内置的机器学习 API ([WinML](https://learn.microsoft.com/en-us/windows/ai/windows-ml/overview))。

## 使用方法

在 Releases 中下载程序与 PP-OCRv5.zip，然后将 PP-OCRv5.zip 中的内容解压到程序同目录下的 models 文件夹中。

## 限制

* 由于算子限制，需要 Win11 24H2+ 系统。
* 不支持调用 NPU（需要 FP32 计算单元；但仍会出现在设备选择列表中）。