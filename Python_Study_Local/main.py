import os
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import joblib
import librosa
import numpy as np
import pandas as pd
import pyworld as pw
import soundfile as sf

# 获取资源根目录（支持 PyInstaller 打包后的路径）
if getattr(sys, 'frozen', False):
    _ROOT = sys._MEIPASS
else:
    _ROOT = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = r"D:\new_document\Document\voice\voice_xgb_model.pkl"
FEATURES_PATH = r"D:\new_document\Document\voice\voice_feature_names.pkl"
LABELS_PATH = r"D:\new_document\Document\voice\voice_label_mapping.pkl"

_model = None
_feature_names = None
_label_mapping = None
_model_lock = threading.Lock()

HTML_PATH = os.path.join(_ROOT, "voice_clarify.html")

# 训练数据标签映射说明：LabelEncoder按字母序 female→0, male→1
GENDER_MAP = {"男性": "男性", "女性": "女性", "male": "男性", "female": "女性"}

# 训练数据参考范围（用于调试对比）
# 格式: (男性mean, 女性mean, 全局min, 全局max)
TRAINING_REF = {
    "meanfreq": (0.171, 0.191, 0.039, 0.251),
    "sd": (0.065, 0.049, 0.018, 0.115),
    "median": (0.175, 0.196, 0.011, 0.261),
    "Q25": (0.116, 0.165, 0.000, 0.247),
    "Q75": (0.226, 0.223, 0.043, 0.274),
    "IQR": (0.111, 0.058, 0.015, 0.252),
    "skew": (3.30, 2.98, 0.14, 34.73),
    "kurt": (48.3, 24.8, 2.1, 1309.6),
    "sp.ent": (0.917, 0.873, 0.739, 0.982),
    "sfm": (0.472, 0.345, 0.037, 0.843),
    "mode": (0.152, 0.179, 0.000, 0.280),
    "centroid": (0.171, 0.191, 0.039, 0.251),
    "meanfun": (0.116, 0.170, 0.056, 0.238),
    "minfun": (0.034, 0.039, 0.010, 0.204),
    "maxfun": (0.254, 0.264, 0.103, 0.279),
    "meandom": (0.729, 0.930, 0.008, 2.958),
    "mindom": (0.040, 0.065, 0.005, 0.459),
    "maxdom": (4.358, 5.736, 0.008, 21.867),
    "dfrange": (4.318, 5.671, 0.000, 21.844),
    "modindx": (0.177, 0.170, 0.000, 0.932),
}


def _load_model():
    """延迟加载模型，避免启动时长时间等待"""
    global _model, _feature_names, _label_mapping
    if _model is not None:
        return _model, _feature_names, _label_mapping
    with _model_lock:
        if _model is not None:
            return _model, _feature_names, _label_mapping
        _model = joblib.load(MODEL_PATH)
        _feature_names = joblib.load(FEATURES_PATH)
        _label_mapping = joblib.load(LABELS_PATH)
        return _model, _feature_names, _label_mapping


# 全局状态：用于平滑预测结果，防止性别识别震荡
_prediction_history = {
    "last_gender": None,
    "consecutive_count": 0,
    "last_update_time": 0,
    "buffer": []  # 存储最近几次的预测结果 (gender, probability)
}


app = FastAPI(title="声音男女识别 API")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(HTML_PATH, encoding="utf-8") as f:
        return f.read()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _skew(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 4))


def _spectral_entropy(magnitude: np.ndarray) -> float:
    if magnitude.size == 0:
        return 0.0
    power = magnitude ** 2
    power_sum = np.sum(power, axis=0, keepdims=True)
    power_sum[power_sum == 0] = 1.0
    prob = power / power_sum
    entropy = -np.sum(prob * np.log2(prob + 1e-12), axis=0)
    return float(np.mean(entropy))


def _ensure_min_duration(y: np.ndarray, sr: int, min_duration: float = 1.2) -> np.ndarray:
    min_samples = int(sr * min_duration)
    if y.size >= min_samples:
        return y
    if y.size == 0:
        return np.zeros(min_samples, dtype=np.float32)
    pad_total = min_samples - y.size
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="edge")


def extract_audio_features(y: np.ndarray, sr: int, feature_names: list) -> pd.DataFrame:
    """
    提取音频特征，并归一化以匹配训练数据格式。

    训练数据的特征分布（来源于 sr=22050 的音频）：
    - 谱特征（centroid, meanfreq 等）: 除以 sr/2 归一化到 0~1
    - 主频特征（meandom, mindom 等）: 单位为 Hz/500
    - 基频特征（meanfun 等）: 除以 1000 归一化
    - 谱熵: 除以 log2(频点数) 归一化到 0~1
    """
    features: Dict[str, float] = {}
    nyquist = sr / 2.0
    n_fft = 2048
    n_bins = n_fft // 2 + 1

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).ravel()
    # 归一化谱质心相关特征：除以 Nyquist 频率
    features["meanfreq"] = _safe_mean(centroid) / nyquist
    features["sd"] = _safe_std(centroid) / nyquist
    features["median"] = (float(np.median(centroid)) / nyquist) if centroid.size else 0.0
    features["Q25"] = (float(np.percentile(centroid, 25)) / nyquist) if centroid.size else 0.0
    features["Q75"] = (float(np.percentile(centroid, 75)) / nyquist) if centroid.size else 0.0
    features["IQR"] = features["Q75"] - features["Q25"]
    features["skew"] = _skew(centroid)
    features["kurt"] = _kurtosis(centroid)

    flatness = librosa.feature.spectral_flatness(y=y).ravel()
    features["sfm"] = _safe_mean(flatness)

    stft = np.abs(librosa.stft(y, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # 归一化谱熵
    features["sp.ent"] = _spectral_entropy(stft) / np.log2(n_bins)

    if stft.size:
        mean_spectrum = np.mean(stft, axis=1)
        mode_idx = int(np.argmax(mean_spectrum))
        features["mode"] = float(freqs[mode_idx]) / nyquist
        dom_idx = np.argmax(stft, axis=0)
        dom_freq = freqs[dom_idx]
    else:
        features["mode"] = 0.0
        dom_freq = np.array([])

    features["centroid"] = features["meanfreq"]  # 同一值
    # 主频特征转为训练数据单位（Hz/500）以匹配 meandom~0.01-5.0, maxdom~0.01-22
    features["meandom"] = _safe_mean(dom_freq) / 500.0
    features["mindom"] = _safe_min(dom_freq) / 500.0
    features["maxdom"] = _safe_max(dom_freq) / 500.0
    features["dfrange"] = features["maxdom"] - features["mindom"]

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[np.isfinite(f0)]
    except Exception:
        f0 = np.array([])

    # 基频归一化：除以 1000（匹配训练数据 meanfun ~0.06-0.24）
    features["meanfun"] = _safe_mean(f0) / 1000.0
    features["minfun"] = _safe_min(f0) / 1000.0
    features["maxfun"] = _safe_max(f0) / 1000.0
    features["modindx"] = float(np.std(f0) / _safe_mean(f0)) if _safe_mean(f0) > 0 and f0.size else 0.0

    for name in feature_names:
        if name not in features:
            features[name] = np.nan

    return pd.DataFrame([features])[feature_names]


def _smooth_prediction(current_gender: str, current_prob: float) -> tuple[str, float]:
    """
    简单多数投票平滑：最近 5 次中出现次数最多的性别即为输出。
    超过 5 秒未预测则重置。
    """
    global _prediction_history

    now = time.time()

    # 超过 5 秒未预测，重置历史
    if now - _prediction_history["last_update_time"] > 5.0:
        _prediction_history["buffer"] = []

    _prediction_history["last_update_time"] = now
    _prediction_history["buffer"].append((current_gender, current_prob))

    # 保留最近 5 次
    if len(_prediction_history["buffer"]) > 5:
        _prediction_history["buffer"].pop(0)

    # 统计多数
    gender_counts = {}
    total_prob = 0.0
    for g, p in _prediction_history["buffer"]:
        gender_counts[g] = gender_counts.get(g, 0) + 1
        total_prob += p

    most_common = max(gender_counts, key=gender_counts.get)
    avg_prob = total_prob / len(_prediction_history["buffer"])

    return most_common, avg_prob


def _trim_silence(y: np.ndarray, sr: int, top_db: int = 28) -> np.ndarray:
    """裁剪首尾静音，避免 WORLD 在无声段产生杂音"""
    if y.size == 0:
        return y
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed if y_trimmed.size > 0 else y


def _clamp_control(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(np.clip(value, lower, upper))


def _compute_f0_ratio(mean_f0: float, target: str, strength: float = 0.55) -> float:
    strength = _clamp_control(strength)
    if mean_f0 <= 0:
        if target == "female":
            return 1.40 + 0.35 * strength
        elif target == "male":
            return 0.82 - 0.15 * strength
        return 1.0
    if target == "female":
        target_f0 = 210.0 + 80.0 * strength
        ratio = target_f0 / mean_f0
        min_ratio = 0.95
        max_ratio = 1.50 + 0.55 * strength
        return float(np.clip(ratio, min_ratio, max_ratio))
    if target == "male":
        target_f0 = 130.0 - 45.0 * strength
        ratio = target_f0 / mean_f0
        min_ratio = 0.85 - 0.30 * strength
        max_ratio = 1.05
        return float(np.clip(ratio, min_ratio, max_ratio))
    return 1.0


def _compute_formant_factor(target: str, f0_ratio: float, strength: float = 0.55) -> float:
    strength = _clamp_control(strength)
    if target == "female":
        base = 1.08 + 0.10 * strength
        power = 0.22 + 0.08 * strength
        min_factor = 0.80
        max_factor = 1.22 + 0.14 * strength
    elif target == "male":
        base = 0.95 - 0.12 * strength
        power = 0.35 + 0.10 * strength
        min_factor = 0.78 - 0.15 * strength
        max_factor = 1.20
    else:
        base = 1.0
        power = 0.30
        min_factor = 0.75
        max_factor = 1.30
    return float(np.clip(base * (f0_ratio ** power), min_factor, max_factor))


def _warp_spectral_envelope(sp: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """频域拉伸共振峰，混入原声保留音色，避免过度扭曲"""
    if factor == 1.0:
        return sp
    n_frames, n_bins = sp.shape
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    src_freqs = freqs / factor
    warped = np.empty_like(sp)
    for idx in range(n_frames):
        warped[idx] = np.interp(freqs, src_freqs, sp[idx], left=sp[idx, 0], right=sp[idx, -1])
    # 混入 30% 原声，保留自然音色
    blend = 0.30 + 0.40 * abs(factor - 1.0)
    blend = min(blend, 0.70)
    return sp * (1.0 - blend) + warped * blend


def _apply_spectral_tilt(sp: np.ndarray, target: str, brightness: float = 0.55) -> np.ndarray:
    brightness = _clamp_control(brightness)
    if target == "female":
        low_gain = 0.98 - 0.05 * brightness
        high_gain = 1.03 + 0.10 * brightness
        tilt = np.linspace(low_gain, high_gain, sp.shape[1], dtype=np.float64)
        return sp * tilt
    if target == "male":
        low_gain = 1.00 + 0.05 * brightness
        high_gain = 0.99 - 0.06 * brightness
        tilt = np.linspace(low_gain, high_gain, sp.shape[1], dtype=np.float64)
        return sp * tilt
    return sp


def _mix_with_original(sp_original: np.ndarray, sp_converted: np.ndarray, target: str, strength: float) -> np.ndarray:
    """频率自适应的频谱混合：低频保留原声（保持音色），高频使用转换（调整音高感受）"""
    strength = _clamp_control(strength)
    n_bins = sp_original.shape[1]

    mix_base = np.linspace(0.20, 0.65, n_bins, dtype=np.float64)
    mix_ratio = mix_base + strength * 0.25

    if target == "female":
        mix_ratio += 0.03
    elif target == "male":
        mix_ratio -= 0.02

    mix_ratio = np.clip(mix_ratio, 0.18, 0.85)
    return sp_original * (1.0 - mix_ratio) + sp_converted * mix_ratio


def _smooth_f0(f0: np.ndarray, window: int = 7) -> np.ndarray:
    """三阶段F0平滑：中值去野 → 低通消阶梯 → 插值填空洞，减少电音"""
    if window <= 1:
        return f0
    half = window // 2
    smoothed = f0.copy()

    # 阶段1：中值滤波（移除离群值），窗口加大到7
    for idx in range(f0.size):
        start = max(0, idx - half)
        end = min(f0.size, idx + half + 1)
        voiced = f0[start:end]
        voiced = voiced[voiced > 0]
        if voiced.size:
            smoothed[idx] = float(np.median(voiced))

    # 阶段2：有声段移动平均低通，消除中值滤波的阶梯状（机械电音主因）
    voiced_mask = smoothed > 0
    if np.any(voiced_mask):
        smoothed_ma = smoothed.copy()
        for idx in range(smoothed.size):
            start = max(0, idx - 2)
            end = min(smoothed.size, idx + 3)
            neighborhood = smoothed[start:end]
            voiced_neighbors = neighborhood[neighborhood > 0]
            if voiced_neighbors.size >= 2:
                smoothed_ma[idx] = float(np.mean(voiced_neighbors))
        # 只在有声段应用低通结果
        smoothed[voiced_mask] = smoothed_ma[voiced_mask]

    # 阶段3：局部线性插值（填充短空洞，扩大范围到10帧）
    voiced_indices = np.where(smoothed > 0)[0]
    if len(voiced_indices) > 1:
        for seg_start_idx in range(len(voiced_indices) - 1):
            idx_a = voiced_indices[seg_start_idx]
            idx_b = voiced_indices[seg_start_idx + 1]
            if idx_b - idx_a <= 10:
                smoothed[idx_a:idx_b + 1] = np.linspace(
                    smoothed[idx_a], smoothed[idx_b], idx_b - idx_a + 1)

    return smoothed


def _smooth_spectral_envelope(sp: np.ndarray, window: int = 5) -> np.ndarray:
    """两步频谱平滑，保留清晰度同时减少噪声"""
    if window <= 1:
        return sp
    
    # 第一阶段：频率维度平滑（倒谱平滑）
    log_sp = np.log(sp + 1e-12)
    kernel = np.ones(window, dtype=np.float64) / window
    
    smoothed = np.empty_like(log_sp)
    for idx in range(log_sp.shape[0]):
        smoothed[idx] = np.convolve(log_sp[idx], kernel, mode="same")
    
    # 第二阶段：时间维度平滑（避免帧间的剧烈跳变）
    if smoothed.shape[0] > 1:
        kernel_time = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        smoothed_time = np.zeros_like(smoothed)
        smoothed_time[0] = smoothed[0]
        for frame_idx in range(1, smoothed.shape[0] - 1):
            smoothed_time[frame_idx] = (
                0.25 * smoothed[frame_idx - 1] +
                0.5 * smoothed[frame_idx] +
                0.25 * smoothed[frame_idx + 1]
            )
        smoothed_time[-1] = smoothed[-1]
        smoothed = smoothed_time
    
    return np.exp(smoothed)


def _adjust_aperiodicity(ap: np.ndarray, f0: np.ndarray, f0_ratio: float, target: str) -> np.ndarray:
    """仅调整有声帧的非周期性，无声帧保持原样（避免呼吸/尾音被合成成杂音）"""
    adjusted = ap.copy()
    n_bins = ap.shape[1]
    voiced_mask = f0 > 0  # 仅处理有声帧

    if not np.any(voiced_mask):
        return adjusted

    if target == "female":
        intensity = float(np.clip((f0_ratio - 1.0) / 1.2, 0.0, 1.0))
        hf_boost = 1.0 + 0.06 * intensity
        hf_mask = np.linspace(1.0, hf_boost, n_bins)
        adjusted[voiced_mask] = ap[voiced_mask] * hf_mask
    elif target == "male":
        intensity = float(np.clip((1.0 - f0_ratio) / 0.6, 0.0, 1.0))
        hf_cut = 1.0 - 0.06 * intensity
        hf_mask = np.linspace(1.0, hf_cut, n_bins)
        adjusted[voiced_mask] = ap[voiced_mask] * hf_mask

    return np.clip(adjusted, 0.0, 1.0)


def _detect_voice_activity(y: np.ndarray, sr: int, min_rms: float = 0.005) -> bool:
    """快速语音活动检测（RMS 能量），阈值 0.005 适应多数麦克风"""
    if y.size == 0:
        return False
    rms = float(np.sqrt(np.mean(y ** 2)))
    return rms >= min_rms


def _extract_voiced_segments(y: np.ndarray, sr: int, frame_length: int = 256, hop_length: int = 128) -> np.ndarray:
    """
    提取音频中的有效人声片段，过滤杂音
    
    Args:
        y: 原始音频信号
        sr: 采样率
        frame_length: 帧长度（样本点数）
        hop_length: 帧移（样本点数）
    
    Returns:
        np.ndarray: 过滤后的音频信号（只保留人声部分）
    """
    try:
        # 使用harvest获取基频
        f0, timeaxis = pw.harvest(y, sr, f0_floor=50.0, f0_ceil=500.0)
        
        # 将F0时间轴转换为样本索引
        f0_times = timeaxis
        f0_indices = (f0_times * sr).astype(int)
        
        # 创建掩码，标记有效人声区域
        voiced_mask = np.zeros(len(y), dtype=bool)
        
        # 对每个有效F0点，标记其周围的样本为人声
        for i, f0_val in enumerate(f0):
            if f0_val > 0:  # 有效人声
                center_idx = f0_indices[i]
                # 标记前后各2个帧长的区域
                start_idx = max(0, center_idx - 2 * frame_length)
                end_idx = min(len(y), center_idx + 2 * frame_length)
                voiced_mask[start_idx:end_idx] = True
        
        # 如果没有人声区域，返回原始音频（避免完全静音）
        if not np.any(voiced_mask):
            return y
            
        # 应用掩码，保留人声部分，其他部分设为接近0的小值
        y_filtered = y.copy()
        y_filtered[~voiced_mask] = 0.0
        
        return y_filtered
    except Exception:
        # 如果过滤失败，返回原始音频
        return y


@app.post("/predict_voice")
async def predict_voice(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        temp_audio_path = temp_file.name

    try:
        y, sr = librosa.load(temp_audio_path, sr=22050, mono=True)
        duration = float(y.size / sr) if sr else 0.0
        
        # 新增：人声活动检测
        if not _detect_voice_activity(y, sr):
            # 如果检测到主要是杂音，返回低置信度结果
            return {
                "status": "success",
                "gender": "unknown",
                "confidence": 0.0,
                "display_features": {
                    "mean_frequency": 0.0,
                    "mean_pitch": 0.0,
                    "amplitude": round(float(np.max(np.abs(y))), 4) if y.size else 0.0,
                    "duration": round(duration, 3),
                },
                "message": "未检测到足够的人声，请确保在安静环境中清晰说话"
            }
        
        # 延迟加载模型（首次调用时加载）
        model, feature_names, label_mapping = _load_model()

        # 确保音频最小时长（仅对过短音频做边缘填充）
        y_for_feature = _ensure_min_duration(y, sr, min_duration=1.2)
        feature_df = extract_audio_features(y_for_feature, sr, feature_names)

        raw_proba = model.predict_proba(feature_df)[0]
        pred_prob = float(raw_proba.max())
        pred_label = int(model.predict(feature_df)[0])
        raw_gender_cn = str(label_mapping.get(pred_label, pred_label))
        predicted_gender = GENDER_MAP.get(raw_gender_cn, raw_gender_cn)

        # 幅度取最后 1 秒，避免累积音频导致图表变平
        recent_y = y[-int(sr):] if y.size > sr else y

        display_features = {
            "mean_frequency": round(float(feature_df["meanfreq"].iloc[0]), 3),
            "mean_pitch": round(float(feature_df["meanfun"].iloc[0]), 3),
            "amplitude": round(float(np.max(np.abs(recent_y))), 4) if recent_y.size else 0.0,
            "duration": round(duration, 3),
        }

        # 收集所有 20 个特征值用于调试
        all_features = {}
        for col in feature_names:
            val = float(feature_df[col].iloc[0])
            all_features[col] = round(val, 6) if not np.isnan(val) else "NaN"

        # 找出偏离训练数据范围的特征
        outliers = []
        for col in feature_names:
            if col in TRAINING_REF and isinstance(all_features[col], float):
                _, _, ref_min, ref_max = TRAINING_REF[col]
                val = all_features[col]
                if val < ref_min * 0.5 or val > ref_max * 1.5:
                    outliers.append(f"{col}={val:.4f}(ref:[{ref_min:.4f},{ref_max:.4f}])")

        return {
            "status": "success",
            "gender": predicted_gender,
            "confidence": round(float(pred_prob * 100), 2),
            "display_features": display_features,
            "debug": {
                "raw_prediction": predicted_gender,
                "model_proba": f"{raw_proba[0]:.4f}/{raw_proba[1]:.4f}",
                "features": all_features,
                "outliers": outliers,
                "duration": round(duration, 3),
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/convert_voice")
async def convert_voice(
    file: UploadFile = File(...),
    target: str = Form("female"),
    fem_strength: float = Form(0.55),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        temp_audio_path = temp_file.name

    try:
        y, sr = librosa.load(temp_audio_path, sr=22050, mono=True)
        y = y.astype(np.float64)
        # 裁剪首尾静音，避免 WORLD 在无声段产生杂音和尾音
        y = _trim_silence(y, sr)
        f0, t = pw.harvest(y, sr, f0_floor=50.0, f0_ceil=500.0)
        f0 = pw.stonemask(y, f0, t, sr)
        sp = pw.cheaptrick(y, f0, t, sr)
        ap = pw.d4c(y, f0, t, sr)

        voiced_f0 = f0[f0 > 0]
        mean_f0 = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
        fem_strength = _clamp_control(fem_strength)
        brightness = 0.35 + fem_strength * 0.45
        f0_ratio = _compute_f0_ratio(mean_f0, target, fem_strength)
        formant_factor = _compute_formant_factor(target, f0_ratio, fem_strength)

        f0_converted = _smooth_f0(f0 * f0_ratio)
        sp_converted = _warp_spectral_envelope(sp, sr, formant_factor)
        sp_converted = _apply_spectral_tilt(sp_converted, target, brightness)
        sp_converted = _mix_with_original(sp, sp_converted, target, fem_strength)
        sp_converted = _smooth_spectral_envelope(sp_converted, window=3)
        ap_converted = _adjust_aperiodicity(ap, f0_converted, f0_ratio, target)
        y_converted = pw.synthesize(f0_converted, sp_converted, ap_converted, sr)

        if y_converted.size:
            peak = float(np.max(np.abs(y_converted)))
            if peak > 1.0:
                y_converted = y_converted / peak

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_file:
            sf.write(out_file.name, y_converted.astype(np.float32), sr)
            return FileResponse(out_file.name, media_type="audio/wav", filename="converted.wav")
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    def open_browser():
        time.sleep(1.0)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    # 后台预热：提前加载模型，避免首次请求等待
    def warmup():
        try:
            _load_model()
        except Exception:
            pass

    threading.Thread(target=warmup, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)