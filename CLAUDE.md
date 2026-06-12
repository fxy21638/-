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

# 训练模型（旧版 — 随机划分）
cd Python_Study_Local && python voice_clarify_train.py

# 训练模型（新版 — 演员独立划分，Actor 1-22 训练，23-24 测试）
cd Python_Study_Local && python voice_clarify_train_v2.py

# 训练并导出模型文件
cd Python_Study_Local && python voice_clarify_mode.py

# 打包 EXE
python build_gui_exe.py   # GUI 版
python build_exe.py       # Web 版

# 诊断脚本（调试用，不参与构建）
cd Python_Study_Local && python debug_scripts/diagnose_conversion.py
```

## 架构核心

**`main.py` 是算法层，`voice_gender_gui.py` 是 GUI 层，通过 import 共享代码：**

```python
# voice_gender_gui.py 从 main.py 导入纯函数
from main import (
    extract_audio_features, _ensure_min_duration, _trim_silence,
    _smooth_f0, _shift_formant_envelope,
    _apply_spectral_tilt, _mix_with_original, _clamp_control,
    _compute_f0_ratio, _adjust_aperiodicity,
    _post_process, predict_gender_segmented,
)
```

这个导入关系意味着：**修改 `main.py` 中的变声/特征提取函数会影响两个界面**。GUI 不启动 FastAPI，只在本地直接调用这些函数。

**关键函数调用链（变声 — 男→女）：**
1. `ConvertWorker.run()` → pyworld 提取 F0/SP/AP
2. `_compute_f0_ratio()` → 计算基频缩放比（`target_f0=200+60*strength`, `max_ratio=1.80+0.80*strength`）
3. `_smooth_f0(f0 * f0_ratio)` → 两阶段平滑：有声帧中值滤波 → 短间隙线性插值
4. `_shift_formant_envelope(sp, sr, target, strength)` → 基于共振峰检测的局部频域拉伸
5. `_apply_spectral_tilt(sp, target, brightness)` → 多段非线性频谱整形
6. `_mix_with_original(sp_original, sp_converted, target, strength)` → 频谱混合（flat 85-91%）
7. `_adjust_aperiodicity(ap, f0, f0_ratio, target)` → 非周期性调整（仅处理有声帧 f0>0）
8. `pw.synthesize()` → WORLD 合成
9. `_post_process(y, sr)` → 7.5kHz 低通滤波

**变声参数（main.py，最新值）：**
| 参数 | 男→女 | 女→男 |
|------|-------|-------|
| F0 目标 | `200+60*strength` Hz | `120-55*strength` Hz |
| F0 比上限 | `1.80+0.80*strength` | `1.05` |
| 共振峰移位比 | `1.18+0.15*strength` (18-33%) | `0.90-0.10*strength` (10-20%) |
| 共振峰混合比 | `0.50+0.25*strength` (50-75%) | 同左 |
| 频谱倾斜 | 削低频(50-220Hz) + 中高频微调 | 增强低频(50-250Hz) + 大幅削高频 |
| 频谱混合 | linspace(0.85, 0.91) + 0.06*strength | 同左 |

**共振峰移位原理（`_shift_formant_envelope`）：**
1. `_estimate_formant_peaks()` — 在 log 域用 `argrelextrema` 检测每帧的 F1-F4 共振峰
2. 对每帧每个共振峰的 ±30% 频宽区域做局部频域拉伸（`np.interp` 重采样）
3. 混合比 `0.50+0.25*strength` 将拉伸后的频谱与原始在该区域内混合
4. 若 <10% 帧检出共振峰，回退到全局 warping（`_warp_spectral_envelope_fallback`）

**`_trim_silence()`** — 用 `librosa.effects.trim(top_db=28)` 裁掉首尾静音。仅在变声流程中 WORLD 分析前调用，避免无声段产生杂音。**注意：预测流程中不使用**——3秒窗口已够短，裁切后 `_ensure_min_duration` 填充会产生畸形特征导致误判。

**关键函数调用链（预测）：**
1. `PredictWorker.run()` → VAD 检测（active_ratio > 10%）→ 硬性峰值检查（>0.008）
2. `predict_gender_segmented()` → 全局特征预测性别 + 2.5s窗口分段投票计算男女声占比
3. `_load_model()` → `extract_audio_features()` → XGBoost 推理 → 显示结果
4. 文件上传显示男女声占比（ratio_label），实时麦克风不显示（始终同一个人）

**GUI 定时器架构（voice_gender_gui.py）：**
- `_chart_timer` (100ms) → `_tick_chart()`: 取 0.4s 音频，更新幅度走势图（800点）+ 实时频谱线图（8192点FFT），若开启预览则每 300ms 运行 WORLD 转换管线
- `_predict_timer` (800ms) → `_tick_predict()`: 取 3s 音频，启动 `PredictWorker` 线程，更新性别标签和调试信息
- 图表自启动后立即绘制，不再等待首次预测完成
- 实时频谱预览：橙色=原始频谱，青色=经 WORLD 转换后的频谱，用于调试变声参数

**频谱显示参数（voice_gender_gui.py: `_update_spectrum`）：**
- 8192 点 FFT（2.7Hz/bin 分辨率），Hann 窗，取最近 0.4s 音频
- 对数频率轴 50-5000Hz，纵轴 -40~0 dB
- `ref=np.max(spec)` 频域自身最大值归一化（非时域峰值）
- 5 点移动平均平滑，保留谐波峰结构
- EMA 噪声门限（α=0.15），底噪压平为稳定水平线
- fill_between 填充显示

## 模型信息

**训练方式：** 演员独立划分（Actor 1-22 训练，Actor 23-24 测试），避免数据泄漏。

**训练脚本：** `voice_clarify_train_v2.py` — 从 RAVDESS 音频直接提取特征（1320条），可选与 CSV 特征合并训练，选择最优方案保存为 `.pkl` 文件。

**模型性能：**
- 测试集准确率：79.2%（Actor 23-24，264条）
- 男声准确率：61.7%（Actor_23 的 meandom 偏高——录音条件域偏移）
- 女声准确率：96.7%
- 全量数据准确率：95.5%（1320条，Actor 1-24 混合预测）
- meanfun（平均基频）占模型重要性 10.2%，maxfun（最高基频）8.2%，modindx（调制指数）7.3%，Top 3 合计 25.7%（特征分布均衡，不依赖单一特征）

**模型局限：**
- 训练数据仅含 RAVDESS（北美英语演讲），对非英语语音、歌唱、极端音域泛化有限
- 唱歌、假声、吼叫等非正常说话可能导致误判（F0 升高→被误判为女声）
- 手机/扬声器回放识别不准：扬声器低频截止（<400Hz）导致男性基频丢失，频响染色使谱特征偏移（域偏移问题，非代码 bug）
- 频谱熵（sp.ent）极低（<0.5）的音频超出训练分布，模型行为不可预测

**变声效果局限：**
- **女→男：** 100% 成功率（降低 F0 + 削高频 + 增强低频，简单可靠）
- **男→女：** ~50-60% 成功率（需大幅提升 F0 + 共振峰移位，频谱特征可能残留男性特征）
- 男→女转换的瓶颈：当原始音频的频谱特征（sp.ent、maxfun、meandom）为极端男性化时，信号处理无法充分改变这些特征。这是训练数据多样性的限制，不是代码 bug

## 模型文件

三个 `.pkl` 文件位于 `D:\new_document\Document\voice\`（硬编码路径）：
- `voice_xgb_model.pkl` — XGBoost 二分类器（max_depth=3, n_estimators=100）
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
- 变声管线中 `_smooth_spectral_envelope` 会降低频谱熵，实际测试证明对转换效果反作用，已从 GUI 管线中移除
- 调试脚本统一放在 `debug_scripts/` 目录下，已在 `.gitignore` 中排除
