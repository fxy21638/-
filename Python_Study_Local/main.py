from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import librosa
import numpy as np
import pandas as pd
import tempfile
from typing import Dict

# 1. 加载预训练模型、特征列表、标签映射
model = joblib.load(r"D:\new_document\Document\voice\voice_xgb_model.pkl")
feature_names = joblib.load(r"D:\new_document\Document\voice\voice_feature_names.pkl")
label_mapping = joblib.load(r"D:\new_document\Document\voice\voice_label_mapping.pkl")

# 2. 初始化FastAPI应用（解决跨域问题）
app = FastAPI(title="声音男女识别API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段允许所有前端域名，生产环境需指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 定义特征提取函数（关键：与训练时特征完全一致）
def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _safe_min(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.min(values))


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


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
    power = magnitude**2
    power_sum = np.sum(power, axis=0, keepdims=True)
    power_sum[power_sum == 0] = 1.0
    prob = power / power_sum
    entropy = -np.sum(prob * np.log2(prob + 1e-12), axis=0)
    return float(np.mean(entropy))


def extract_audio_features(y: np.ndarray, sr: int) -> pd.DataFrame:
    """
    从音频文件中提取声学特征
    :param audio_path: 音频文件路径（WAV格式）
    :return: 特征DataFrame（与模型输入一致）
    """
    features: Dict[str, float] = {}

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).ravel()
    features['meanfreq'] = _safe_mean(centroid)
    features['sd'] = _safe_std(centroid)
    features['median'] = float(np.median(centroid)) if centroid.size else 0.0
    features['Q25'] = float(np.percentile(centroid, 25)) if centroid.size else 0.0
    features['Q75'] = float(np.percentile(centroid, 75)) if centroid.size else 0.0
    features['IQR'] = features['Q75'] - features['Q25']

    features['skew'] = _skew(centroid)
    features['kurt'] = _kurtosis(centroid)

    flatness = librosa.feature.spectral_flatness(y=y).ravel()
    features['sfm'] = _safe_mean(flatness)

    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    features['sp.ent'] = _spectral_entropy(stft)

    if stft.size:
        mean_spectrum = np.mean(stft, axis=1)
        mode_idx = int(np.argmax(mean_spectrum))
        features['mode'] = float(freqs[mode_idx])
    else:
        features['mode'] = 0.0

    features['centroid'] = _safe_mean(centroid)

    if stft.size:
        dom_idx = np.argmax(stft, axis=0)
        dom_freq = freqs[dom_idx]
    else:
        dom_freq = np.array([])
    features['meandom'] = _safe_mean(dom_freq)
    features['mindom'] = _safe_min(dom_freq)
    features['maxdom'] = _safe_max(dom_freq)
    features['dfrange'] = features['maxdom'] - features['mindom']

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[np.isfinite(f0)]
    except Exception:
        f0 = np.array([])
    features['meanfun'] = _safe_mean(f0)
    features['minfun'] = _safe_min(f0)
    features['maxfun'] = _safe_max(f0)
    if features['meanfun'] > 0:
        features['modindx'] = float(np.std(f0) / features['meanfun']) if f0.size else 0.0
    else:
        features['modindx'] = 0.0

    for name in feature_names:
        if name not in features:
            features[name] = np.nan
    
    # 转换为DataFrame（模型要求输入格式）
    feature_df = pd.DataFrame([features])[feature_names]  # 强制按feature_names排序
    return feature_df

# 4. 定义API端点（接收音频，返回预测结果和特征）
@app.post("/predict_voice")
async def predict_voice(file: UploadFile = File(...)):
    """
    接收前端上传的音频文件（WAV格式），返回识别结果和关键特征
    """
    # 保存上传的音频文件到临时路径
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        temp_audio_path = temp_file.name
    
    try:
        # 提取特征
        y, sr = librosa.load(temp_audio_path, sr=22050, mono=True)
        feature_df = extract_audio_features(y, sr)
        # 模型预测（概率和类别）
        pred_prob = float(model.predict_proba(feature_df)[0].max())  # 最大置信度
        pred_label = int(model.predict(feature_df)[0])              # 预测标签（0/1）
        pred_gender = str(label_mapping.get(pred_label, pred_label))  # 转换为性别（男/女）
        
        # 提取用于前端显示的关键特征（幅度、频率）
        display_features = {
            "mean_frequency": round(float(feature_df['meanfreq'].iloc[0]), 2),  # 平均频率
            "amplitude": round(float(np.max(np.abs(y))), 4) if y.size else 0.0,
        }
        
        # 返回结果
        return {
            "status": "success",
            "gender": pred_gender,
            "confidence": round(float(pred_prob * 100), 2),  # 置信度（%）
            "display_features": display_features
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 5. 运行服务（终端执行：uvicorn main:app --reload）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)