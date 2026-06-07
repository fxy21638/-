# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

声音性别识别与变声项目，基于 XGBoost + pyworld 声码器。两种界面共享同一套算法核心。

## 常用命令

```bash
# 启动桌面 GUI（主要开发目标）
cd Python_Study_Local && python voice_gender_gui.py

# 启动 Web 后端
cd Python_Study_Local && python main.py

# 训练模型
cd Python_Study_Local && python voice_clarify_train.py

# 训练并导出模型文件
cd Python_Study_Local && python voice_clarify_mode.py

# 打包 EXE
python build_gui_exe.py   # GUI 版
python build_exe.py       # Web 版
```

## 架构核心

**`main.py` 是算法层，`voice_gender_gui.py` 是 GUI 层，通过 import 共享代码：**

```python
# voice_gender_gui.py 从 main.py 导入纯函数
from main import (
    extract_audio_features, _ensure_min_duration, _trim_silence,
    _smooth_f0, _smooth_spectral_envelope, _warp_spectral_envelope,
    _apply_spectral_tilt, _mix_with_original, _clamp_control,
    _compute_f0_ratio, _compute_formant_factor, _adjust_aperiodicity,
)
```

这个导入关系意味着：**修改 `main.py` 中的变声/特征提取函数会影响两个界面**。GUI 不启动 FastAPI，只在本地直接调用这些函数。

**关键函数调用链（变声）：**
1. `ConvertWorker.run()` → pyworld 提取 F0/SP/AP
2. `_compute_f0_ratio()` → 计算基频缩放比
3. `_compute_formant_factor()` → 计算共振峰偏移系数
4. `_smooth_f0(f0 * f0_ratio)` → 三阶段平滑：中值去野 → 移动平均低通 → 线性插值填空洞
5. `_warp_spectral_envelope(sp, sr, formant_factor)` → 频域拉伸 + 30-70% 原始包络混合
6. `_apply_spectral_tilt(sp, target, brightness)` → 频率倾斜
7. `_mix_with_original(sp_original, sp_converted, target, strength)` → 频谱混合
8. `_smooth_spectral_envelope(sp, window)` → log 域 + 时域两步平滑
9. `_adjust_aperiodicity(ap, f0, f0_ratio, target)` → 非周期性调整（仅处理有声帧 f0>0）
10. `pw.synthesize()` → WORLD 合成

**`_trim_silence()`** — 用 `librosa.effects.trim(top_db=28)` 裁掉首尾静音。仅在变声流程中 WORLD 分析前调用，避免无声段产生杂音。**注意：预测流程中不使用**——3秒窗口已够短，裁切后 `_ensure_min_duration` 填充会产生畸形特征导致误判。

**关键函数调用链（预测）：**
1. `PredictWorker.run()` → VAD 检测（active_ratio > 10%）→ 硬性峰值检查（>0.008）→ `_ensure_min_duration()` → `_load_model()` → `extract_audio_features()` → XGBoost 推理 → 显示结果

**GUI 定时器架构（voice_gender_gui.py）：**
- `_chart_timer` (100ms) → `_tick_chart()`: 取 0.4s 音频，更新幅度走势图（800点）+ 实时频谱线图（8192点FFT），若开启预览则每 300ms 运行 WORLD 转换管线
- `_predict_timer` (1500ms) → `_tick_predict()`: 取 3s 音频，启动 `PredictWorker` 线程，更新性别标签和调试信息
- 图表自启动后立即绘制，不再等待首次预测完成
- 实时频谱预览：橙色=原始频谱，青色=经 WORLD 转换后的频谱，用于调试变声参数

**频谱显示参数（voice_gender_gui.py: `_update_spectrum`）：**
- 8192 点 FFT（2.7Hz/bin 分辨率），Hann 窗，取最近 0.4s 音频
- 对数频率轴 50-5000Hz，纵轴 -40~0 dB
- `ref=np.max(spec)` 频域自身最大值归一化（非时域峰值）
- 5 点移动平均平滑，保留谐波峰结构
- EMA 噪声门限（α=0.15），底噪压平为稳定水平线
- fill_between 填充显示

**模型局限：**
- 训练数据：RAVDESS 数据集（1320 条语音，660 男 + 660 女）+ 4 种噪音增强（干净/白噪低/白噪中/50Hz 电噪），5 折 CV 准确率 91.1%，测试集准确率 87.5%
- 唱歌、假声、吼叫等非正常说话可能导致误判（F0 升高→被误判为女声）
- 手机/扬声器回放识别不准：扬声器低频截止（<400Hz）导致男性基频丢失，频响染色使谱特征偏移（域偏移问题，非代码 bug）
- meanfun（平均基频）占模型重要性 10.2%，maxfun（最高基频）8.2%，modindx（调制指数）7.3%，Top 3 合计 25.7%（特征分布均衡，不依赖单一特征）

## 模型文件

三个 `.pkl` 文件位于 `D:\new_document\Document\voice\`（硬编码路径）：
- `voice_xgb_model.pkl` — XGBoost 二分类器
- `voice_feature_names.pkl` — 18 个特征名列表
- `voice_label_mapping.pkl` — `{0: "女性", 1: "男性"}`

GUI 打包（`build_gui_exe.py`）时会将模型嵌入 `model/` 子目录，通过 `sys._MEIPASS` 读取。

## 重要约定

- 所有音频统一重采样到 **22050Hz 单声道**
- 特征值已归一化（谱特征 / nyquist，基频 / 1000，主频 / 500），与训练数据分布对齐
- `TRAINING_REF` 字典（main.py:42-63）是训练数据的参考范围，用于调试异常特征检测
- `GENDER_MAP` 只做中英文映射，不做性别转换
- `_clamp_control()` 确保 strength/brightness 参数在 [0, 1] 范围
- `strength` 默认值 0.55，`brightness = 0.35 + strength * 0.45`
