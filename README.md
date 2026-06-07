# Python Study — 声音男女识别

基于 XGBoost + pyworld 的实时声音性别识别与变声项目，提供 Web（FastAPI + HTML）和桌面（PySide6）两种交互方式。

## 功能

- **实时声音性别识别**：麦克风采集音频，每 1.5 秒输出性别预测 + 置信度
- **声音转换**：将录音转换为男声或女声（pyworld F0 + 频谱包络处理）
- **音频文件测试**：上传 WAV/MP3 文件进行离线识别，展示全部 18 个声学特征
- **实时图表**：声音幅度和归一化频率趋势图
- **两种界面**：Web 版（浏览器录音）和桌面版（sounddevice 采集，更稳定）

## 项目结构

```text
Python_Study_Local/
├── main.py                  # FastAPI 后端：预测 + 变声 API
├── voice_gender_gui.py      # PySide6 桌面 GUI（推荐使用）
├── voice_clarify.html       # Web 前端界面
├── voice_clarify_train.py   # 模型训练脚本
├── voice_clarify_mode.py    # 模型保存脚本
├── qt_sinx_gui.py           # sin(x) 绘图 Qt 示例
├── helloworld.py            # Python 基础入门
├── sinx_plot.ipynb          # Jupyter 笔记本
├── build_exe.py             # Web 版打包脚本 → VoiceGender.exe
└── build_gui_exe.py         # 桌面版打包脚本 → VoiceGenderGUI.exe
```

## 快速开始

### 环境要求

- Python 3.12+
- Windows 11（sounddevice 依赖 PortAudio）

### 安装依赖

```bash
pip install fastapi uvicorn PySide6 librosa soundfile pyworld xgboost scikit-learn matplotlib sounddevice
```

### 模型文件

将以下 3 个 .pkl 文件放入 `D:\new_document\Document\voice\` 目录：

| 文件 | 说明 |
| ---- | ---- |
| `voice_xgb_model.pkl` | 训练好的 XGBoost 分类器 |
| `voice_feature_names.pkl` | 18 个声学特征名称列表 |
| `voice_label_mapping.pkl` | 标签映射（0=女性, 1=男性） |

> 模型基于 [RAVDESS 数据集](https://zenodo.org/record/1188976) 训练（1320 条语音 + 4 种噪音增强），5 折 CV 准确率 91.1%。

### 运行桌面版（推荐）

```bash
cd Python_Study_Local
python voice_gender_gui.py
```

### 运行 Web 版

```bash
cd Python_Study_Local
python main.py           # 启动 FastAPI 服务 (端口 8000)
# 浏览器打开 voice_clarify.html（需通过 localhost 访问以使用麦克风）
```

## 桌面版使用说明

1. 启动后等待状态栏显示「模型加载完成 - 就绪」
2. 点击 **开始采集**，对着麦克风说话
3. 右侧图表实时显示声音幅度和频率变化
4. 停止后可 **导出录音 WAV** 或进行 **声音转换**
5. 女声强度滑块可调节转换的明亮度（推荐 45%–65%）

### 打包为 exe

```bash
python build_gui_exe.py
# 输出：dist/VoiceGenderGUI/VoiceGenderGUI.exe
```

## 技术栈

| 层 | 桌面版 | Web 版 |
| --- | --- | --- |
| UI | PySide6 + matplotlib | HTML5 + Chart.js |
| 音频采集 | sounddevice (PortAudio) | Web Audio API |
| 特征提取 | librosa | librosa (后端) |
| 分类模型 | XGBoost (18 特征) | 同左 |
| 声音转换 | pyworld (F0 + 频谱) | 同左 |

## License

用于课程学习与个人研究。
