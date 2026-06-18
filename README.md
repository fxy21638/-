# Python Study — 声音男女识别与变声

基于 XGBoost + pyworld 的实时声音性别识别与变声项目，提供 Web（FastAPI + HTML）和桌面（PySide6）两种交互方式。

## 功能

- **实时声音性别识别**：麦克风采集音频，每 0.8 秒输出性别预测 + 置信度
- **声音转换**：男声↔女声互转（pyworld F0缩放 + 共振峰移位 + 频谱整形）
- **音频文件测试**：上传 WAV/MP3 文件进行离线识别，展示男女声占比
- **实时图表**：声音幅度走势图 + 8192点FFT实时频谱图
- **实时频谱预览**：调试模式下实时显示原始/转换频谱对比（橙色=原始，青色=转换后）
- **两种界面**：Web 版（浏览器录音）和桌面版（sounddevice 采集，更稳定）

## 项目结构

```text
Python_Study_Local/
├── main.py                     # FastAPI 后端 + 算法层：特征提取、VAD、WORLD 变声
├── voice_gender_gui.py         # PySide6 桌面 GUI（推荐使用）
├── voice_clarify.html          # Web 前端界面
├── voice_clarify_train.py      # 模型训练（旧版 — 随机划分）
├── voice_clarify_train_v2.py   # 模型训练（新版 — 演员独立划分）
├── voice_clarify_mode.py       # 模型持久化：训练+导出三个.pkl文件
├── gen_report_figures.py       # 项目报告图表生成
├── build_exe.py                # Web 版打包脚本 → VoiceGender.exe
├── build_gui_exe.py            # 桌面版打包脚本 → VoiceGenderGUI.exe
├── debug_scripts/              # 调试/诊断脚本（不参与构建）
├── requirements.txt            # Python 依赖列表
└── report_figures/             # 项目报告图表
```

## 快速开始

### 环境要求

- Python 3.12+
- Windows 11（sounddevice 依赖 PortAudio）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 模型文件

将以下 3 个 .pkl 文件放入 `D:\new_document\Document\voice\` 目录：

| 文件 | 说明 |
| ---- | ---- |
| `voice_xgb_model.pkl` | 训练好的 XGBoost 分类器 |
| `voice_feature_names.pkl` | 18 个声学特征名称列表 |
| `voice_label_mapping.pkl` | 标签映射（0=女性, 1=男性） |

> 模型基于 [RAVDESS 数据集](https://zenodo.org/record/1188976) 训练（1320 条语音），采用演员独立划分（Actor 1-22 训练，23-24 测试），测试集准确率 79.2%。

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
5. 女声强度滑块可调节转换力度（推荐 45%–65%）
6. **上传文件测试**：导入音频文件，显示性别预测 + 男女声占比
7. **实时预览**：勾选后可在频谱图中实时对比原始/转换后的频谱

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
| 特征提取 | librosa (STFT + YIN) | librosa (后端) |
| 分类模型 | XGBoost (18 特征) | 同左 |
| 声音转换 | pyworld (F0 + 共振峰移位 + 频谱整形) | 同左 |
| 模型训练 | 演员独立划分 (Actor 1-22/23-24) | 同左 |

## 变声效果

| 方向 | 成功率 | 说明 |
|------|--------|------|
| 女→男 | ~100% | 降低F0 + 削高频 + 增强低频，简单可靠 |
| 男→女 | ~50-60% | 需大幅提升F0 + 共振峰移位，极端男声特征难以完全消除 |

> 变声效果取决于原始音频特征。频谱特征（频谱熵、主导频率）极端偏离训练分布的音频，模型可能仍识别为原始性别。这是训练数据多样性的限制，非代码缺陷。

## License

作者：饭香鱼和蓝蓝的天，用于课程学习与个人研究。
有问题请咨询或者提issue，饭香鱼QQ：<2163833350@qq.com>。
