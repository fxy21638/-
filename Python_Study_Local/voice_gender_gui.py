"""声音男女识别 — PySide6 桌面 GUI"""
import os
import sys
import threading
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import pyworld as pw
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 配置 matplotlib 中文字体
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 路径处理：PyInstaller 打包后从 _MEIPASS/model/ 读取 ──────────────
if getattr(sys, "frozen", False):
    _MODEL_DIR = Path(sys._MEIPASS) / "model"
else:
    _MODEL_DIR = Path(r"D:\new_document\Document\voice")

MODEL_PATH = str(_MODEL_DIR / "voice_xgb_model.pkl")
FEATURES_PATH = str(_MODEL_DIR / "voice_feature_names.pkl")
LABELS_PATH = str(_MODEL_DIR / "voice_label_mapping.pkl")

GENDER_MAP = {"男性": "男性", "女性": "女性", "male": "男性", "female": "女性"}

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

# ── 模型加载（独立于 main.py，避免 FastAPI 依赖）───────────────────────
_model = None
_feature_names = None
_label_mapping = None
_model_lock = threading.Lock()


def _load_model():
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


# ── 从 main.py 导入特征提取等纯函数 ────────────────────────────────────
from main import (
    extract_audio_features,
    _ensure_min_duration,
    _smooth_f0,
    _smooth_spectral_envelope,
    _warp_spectral_envelope,
    _apply_spectral_tilt,
    _mix_with_original,
    _clamp_control,
    _compute_f0_ratio,
    _compute_formant_factor,
    _adjust_aperiodicity,
    _trim_silence,
)


# ══════════════════════════════════════════════════════════════════════════
# 音频环形缓冲区（线程安全）
# ══════════════════════════════════════════════════════════════════════════
class AudioRingBuffer:
    def __init__(self, capacity_seconds: float = 15.0, sr: int = 22050):
        self._sr = sr
        self._buffer = np.zeros(int(capacity_seconds * sr), dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total_written

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        with self._lock:
            cap = len(self._buffer)
            if self._write_pos + n <= cap:
                self._buffer[self._write_pos : self._write_pos + n] = data
            else:
                remaining = cap - self._write_pos
                self._buffer[self._write_pos :] = data[:remaining]
                self._buffer[: n - remaining] = data[remaining:]
            self._write_pos = (self._write_pos + n) % cap
            self._total_written += n

    def get_last(self, duration_seconds: float) -> np.ndarray | None:
        n_samples = int(duration_seconds * self._sr)
        with self._lock:
            if self._total_written < n_samples:
                return None
            cap = len(self._buffer)
            start = (self._write_pos - n_samples) % cap
            if start + n_samples <= cap:
                return self._buffer[start : start + n_samples].copy()
            first = self._buffer[start:]
            second = self._buffer[: n_samples - len(first)]
            return np.concatenate([first, second])

    def get_full(self) -> np.ndarray | None:
        n_samples = min(self._total_written, len(self._buffer))
        if n_samples == 0:
            return None
        return self.get_last(float(n_samples) / self._sr)

    def reset(self) -> None:
        with self._lock:
            self._buffer.fill(0.0)
            self._write_pos = 0
            self._total_written = 0


# ══════════════════════════════════════════════════════════════════════════
# 预测工作线程
# ══════════════════════════════════════════════════════════════════════════
class PredictWorker(QThread):
    result_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, audio: np.ndarray, sr: int = 22050, enable_vad: bool = True, parent=None):
        super().__init__(parent)
        self._audio = audio.copy()
        self._sr = sr
        self._enable_vad = enable_vad

    def run(self):
        try:
            y = self._audio.astype(np.float64 if self._audio.dtype == np.float32 else self._audio.dtype)
            duration = float(y.size / self._sr)

            # 人声检测：分帧统计有效语音比例，避免静音/杂音误判
            if self._enable_vad:
                frame_len = int(self._sr * 0.1)
                if frame_len >= 64 and y.size >= frame_len:
                    n_frames = y.size // frame_len
                    frame_rms = np.array([
                        float(np.sqrt(np.mean(y[i * frame_len:(i + 1) * frame_len] ** 2)))
                        for i in range(n_frames)
                    ])
                    active_ratio = float(np.mean(frame_rms > 0.005))
                else:
                    active_ratio = 1.0 if float(np.sqrt(np.mean(y ** 2))) > 0.005 else 0.0

                if active_ratio < 0.10:
                    self.result_ready.emit({
                        "status": "success",
                        "gender": "unknown",
                        "confidence": 0,
                        "display_features": {
                            "mean_frequency": 0.0,
                            "mean_pitch": 0.0,
                            "amplitude": round(float(np.max(np.abs(y))), 4) if y.size else 0.0,
                            "duration": round(duration, 3),
                        },
                        "debug": {"message": "未检测到足够人声"},
                    })
                    return

            model, feature_names, label_mapping = _load_model()
            y_trimmed = _trim_silence(y, self._sr)
            y_for_feature = _ensure_min_duration(y_trimmed, self._sr, min_duration=1.2)
            feature_df = extract_audio_features(y_for_feature, self._sr, feature_names)

            raw_proba = model.predict_proba(feature_df)[0]
            pred_prob = float(raw_proba.max())
            pred_label = int(model.predict(feature_df)[0])
            raw_gender_cn = str(label_mapping.get(pred_label, pred_label))
            predicted_gender = GENDER_MAP.get(raw_gender_cn, raw_gender_cn)

            recent_y = y[-int(self._sr) :] if y.size > self._sr else y

            display_features = {
                "mean_frequency": round(float(feature_df["meanfreq"].iloc[0]), 3),
                "mean_pitch": round(float(feature_df["meanfun"].iloc[0]), 3),
                "amplitude": round(float(np.max(np.abs(recent_y))), 4) if recent_y.size else 0.0,
                "duration": round(duration, 3),
            }

            all_features = {}
            for col in feature_names:
                val = float(feature_df[col].iloc[0])
                all_features[col] = round(val, 6) if not np.isnan(val) else "NaN"

            outliers = []
            for col in feature_names:
                if col in TRAINING_REF and isinstance(all_features[col], float):
                    _, _, ref_min, ref_max = TRAINING_REF[col]
                    val = all_features[col]
                    if val < ref_min * 0.5 or val > ref_max * 1.5:
                        outliers.append(f"{col}={val:.4f}(ref:[{ref_min:.4f},{ref_max:.4f}])")

            # 附加辅助特征到调试信息
            aux_features = {}
            for k in ["fun_range", "freq_fun_ratio", "low_energy_ratio"]:
                if k in feature_df.columns:
                    aux_features[k] = round(float(feature_df[k].iloc[0]), 4)
            result = {
                "status": "success",
                "gender": predicted_gender,
                "confidence": round(pred_prob * 100, 1),
                "display_features": display_features,
                "debug": {
                    "raw_prediction": raw_gender_cn,
                    "model_proba": f"{pred_prob:.4f}",
                    "duration": f"{duration:.1f}s",
                    "features": all_features,
                    "outliers": outliers,
                    "aux_features": aux_features,
                },
            }
            self.result_ready.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════
# 变声工作线程
# ══════════════════════════════════════════════════════════════════════════
class ConvertWorker(QThread):
    finished = Signal(str)   # 输出文件路径
    error_occurred = Signal(str)

    def __init__(self, audio: np.ndarray, sr: int, target: str,
                 strength: float, output_path: str, parent=None):
        super().__init__(parent)
        self._audio = audio.copy()
        self._sr = sr
        self._target = target
        self._strength = strength
        self._output_path = output_path

    def run(self):
        try:
            y = self._audio.astype(np.float64)
            sr = self._sr
            target = self._target
            strength = self._strength

            f0, t = pw.harvest(y, sr, f0_floor=50.0, f0_ceil=500.0)
            f0 = pw.stonemask(y, f0, t, sr)
            sp = pw.cheaptrick(y, f0, t, sr)
            ap = pw.d4c(y, f0, t, sr)

            voiced_f0 = f0[f0 > 0]
            mean_f0 = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
            strength = _clamp_control(strength)
            brightness = 0.35 + strength * 0.45
            f0_ratio = _compute_f0_ratio(mean_f0, target, strength)
            formant_factor = _compute_formant_factor(target, f0_ratio, strength)

            f0_converted = _smooth_f0(f0 * f0_ratio)
            sp_converted = _warp_spectral_envelope(sp, sr, formant_factor)
            sp_converted = _apply_spectral_tilt(sp_converted, target, brightness)
            sp_converted = _mix_with_original(sp, sp_converted, target, strength)
            sp_converted = _smooth_spectral_envelope(sp_converted, window=3)
            ap_converted = _adjust_aperiodicity(ap, f0_converted, f0_ratio, target)
            y_converted = pw.synthesize(f0_converted, sp_converted, ap_converted, sr)

            if y_converted.size:
                peak = float(np.max(np.abs(y_converted)))
                if peak > 1.0:
                    y_converted = y_converted / peak

            sf.write(self._output_path, y_converted.astype(np.float32), sr)
            self.finished.emit(self._output_path)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════
# 主窗口
# ══════════════════════════════════════════════════════════════════════════
class VoiceGenderWindow(QMainWindow):
    MAX_CHART_POINTS = 800
    CHART_INTERVAL_MS = 100
    PREDICT_INTERVAL_MS = 1500
    WINDOW_SECONDS = 3.0
    SAMPLE_RATE = 22050

    _warmup_done = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("声音男女识别")
        self.resize(1050, 700)
        self.setMinimumSize(900, 600)

        # 状态
        self._ring: AudioRingBuffer | None = None
        self._stream: sd.InputStream | None = None
        self._predict_timer: QTimer | None = None
        self._chart_timer: QTimer | None = None
        self._predict_worker: PredictWorker | None = None
        self._convert_worker: ConvertWorker | None = None
        self._upload_worker: PredictWorker | None = None
        self._warmed_up = False
        self._ui_ready = False
        self._full_audio_for_export: np.ndarray | None = None
        self._full_audio_chunks: list[np.ndarray] = []
        self._first_prediction_done = False
        self._amp_history: list[float] = []
        self._line_spec = None
        self._line_conv_spec = None
        self._preview_target: str | None = None  # "female" / "male" / None(关闭预览)
        self._preview_tick = 0
        self._noise_floor = -40.0  # EMA-tracked noise floor for stable baseline clamping
        self._noise_floor_conv = -40.0
        self._unknown_streak = 0       # 连续 unknown 次数防抖
        self._last_displayed_gender = "?"  # 上次显示的性别
        self._gender_streak = 0        # 连续同性别次数，防止男女震荡
        self._pending_gender = None    # 待确认的新性别

        self._setup_ui()
        self._setup_charts()
        self._setup_style()
        self._warmup_done.connect(self._on_warmup_done)
        self._ui_ready = True
        self._update_button_states("idle")

        # 后台加载模型 — 按钮显示加载状态
        self.start_btn.setText("模型加载中...")
        self.statusBar().showMessage("正在加载模型，请稍候...")
        t = threading.Thread(target=self._warmup_model, daemon=True)
        t.start()

    # ── UI 布局 ──────────────────────────────────────────────────────
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # 左面板 ──────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(400)
        left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # 录制控制
        rec_group = QGroupBox("录制控制")
        rec_layout = QVBoxLayout(rec_group)
        rec_layout.setSpacing(6)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.start_btn = QPushButton("开始采集")
        self.start_btn.setMinimumHeight(32)
        self.stop_btn = QPushButton("停止采集")
        self.stop_btn.setMinimumHeight(32)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        rec_layout.addLayout(btn_row)
        self.export_btn = QPushButton("导出录音 WAV")
        self.export_btn.setMinimumHeight(28)
        rec_layout.addWidget(self.export_btn)
        self.rec_status = QLabel("就绪")
        self.rec_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rec_status.setStyleSheet("font-size: 12px; color: #6b7280;")
        rec_layout.addWidget(self.rec_status)
        self.vad_checkbox = QCheckBox("人声检测")
        self.vad_checkbox.setChecked(True)
        rec_layout.addWidget(self.vad_checkbox)
        left_layout.addWidget(rec_group)

        # 变声控制
        conv_group = QGroupBox("声音转换")
        conv_layout = QVBoxLayout(conv_group)
        conv_layout.setSpacing(8)
        strength_row = QHBoxLayout()
        strength_row.setSpacing(8)
        strength_label = QLabel("女声强度:")
        strength_label.setMinimumWidth(60)
        strength_row.addWidget(strength_label)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(55)
        strength_row.addWidget(self.strength_slider)
        self.strength_value_label = QLabel("55%")
        self.strength_value_label.setMinimumWidth(36)
        self.strength_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        strength_row.addWidget(self.strength_value_label)
        conv_layout.addLayout(strength_row)
        conv_btn_row = QHBoxLayout()
        conv_btn_row.setSpacing(8)
        self.to_female_btn = QPushButton("转换为女声")
        self.to_female_btn.setMinimumHeight(30)
        self.to_male_btn = QPushButton("转换为男声")
        self.to_male_btn.setMinimumHeight(30)
        conv_btn_row.addWidget(self.to_female_btn)
        conv_btn_row.addWidget(self.to_male_btn)
        conv_layout.addLayout(conv_btn_row)

        # 实时频谱预览
        preview_row = QHBoxLayout()
        preview_row.setSpacing(6)
        self.preview_female_btn = QPushButton("频谱预览♀")
        self.preview_female_btn.setMinimumHeight(26)
        self.preview_female_btn.setCheckable(True)
        self.preview_female_btn.setStyleSheet("""
            QPushButton { background-color: #e5e7eb; color: #6b7280; border-radius: 5px; font-size: 11px; }
            QPushButton:checked { background-color: #ec4899; color: #fff; font-weight: bold; }
        """)
        self.preview_male_btn = QPushButton("频谱预览♂")
        self.preview_male_btn.setMinimumHeight(26)
        self.preview_male_btn.setCheckable(True)
        self.preview_male_btn.setStyleSheet("""
            QPushButton { background-color: #e5e7eb; color: #6b7280; border-radius: 5px; font-size: 11px; }
            QPushButton:checked { background-color: #3b82f6; color: #fff; font-weight: bold; }
        """)
        preview_row.addWidget(self.preview_female_btn)
        preview_row.addWidget(self.preview_male_btn)
        conv_layout.addLayout(preview_row)
        left_layout.addWidget(conv_group)

        # 文件上传
        upload_group = QGroupBox("文件上传测试")
        upload_layout = QVBoxLayout(upload_group)
        self.upload_btn = QPushButton("选择音频文件测试")
        self.upload_btn.setMinimumHeight(30)
        upload_layout.addWidget(self.upload_btn)
        self.upload_status = QLabel("")
        self.upload_status.setWordWrap(True)
        upload_layout.addWidget(self.upload_status)
        left_layout.addWidget(upload_group)

        # 识别结果
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(4)
        self.gender_label = QLabel("--")
        self.gender_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gender_label.setStyleSheet("font-size: 30px; font-weight: bold; padding: 6px 0; color: #334155;")
        result_layout.addWidget(self.gender_label)
        self.confidence_label = QLabel("置信度：--%")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 15px; color: #b91c1c;")
        result_layout.addWidget(self.confidence_label)
        note = QLabel("※ 当前模型仅支持正常说话声，唱歌/假声/吼叫等可能误判")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet("font-size: 11px; color: #9ca3af; padding-top: 4px;")
        result_layout.addWidget(note)
        left_layout.addWidget(result_group)

        # 调试信息
        debug_group = QGroupBox("调试信息")
        debug_layout = QVBoxLayout(debug_group)
        self.debug_box = QPlainTextEdit()
        self.debug_box.setReadOnly(True)
        self.debug_box.setMaximumHeight(160)
        debug_layout.addWidget(self.debug_box)
        left_layout.addWidget(debug_group)

        left_layout.addStretch()
        main_layout.addWidget(left)

        # 右面板（图表）───────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = FigureCanvas(Figure(figsize=(8, 6), tight_layout=True))
        right_layout.addWidget(self.canvas)
        main_layout.addWidget(right, 1)

        # 信号连接
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.export_btn.clicked.connect(self._on_export)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        self.to_female_btn.clicked.connect(lambda: self._on_convert("female"))
        self.to_male_btn.clicked.connect(lambda: self._on_convert("male"))
        self.preview_female_btn.clicked.connect(lambda checked: self._on_preview_toggle("female", checked))
        self.preview_male_btn.clicked.connect(lambda checked: self._on_preview_toggle("male", checked))
        self.upload_btn.clicked.connect(self._on_upload)

    # ── 图表 ─────────────────────────────────────────────────────────
    def _setup_charts(self):
        self.canvas.figure.set_facecolor("#f7f8fa")
        self.ax_amp = self.canvas.figure.add_subplot(2, 1, 1)
        self.ax_amp.set_facecolor("#ffffff")
        self.ax_amp.set_title("实时声音幅度", fontsize=12, fontweight="bold", color="#374151")
        self.ax_amp.set_ylabel("幅度", fontsize=10, color="#6b7280")
        self.ax_amp.grid(True, alpha=0.2, color="#d1d5db")
        self.ax_amp.set_xlim(0, self.MAX_CHART_POINTS)
        self.ax_amp.set_ylim(0, 0.01)
        self.ax_amp.tick_params(labelsize=9, colors="#6b7280")

        self.ax_spec = self.canvas.figure.add_subplot(2, 1, 2)
        self.ax_spec.set_facecolor("#1a1a2e")
        self.ax_spec.set_title("实时频谱 (Frequency Spectrum)", fontsize=12, fontweight="bold", color="#374151")
        self.ax_spec.set_xlabel("频率 (Hz)", fontsize=10, color="#6b7280")
        self.ax_spec.set_ylabel("幅度 (dB)", fontsize=10, color="#6b7280")
        self.ax_spec.set_xscale("log")
        self.ax_spec.set_xlim(50, 5000)
        self.ax_spec.set_ylim(-40, 0)
        self.ax_spec.tick_params(labelsize=9, colors="#6b7280")
        self.ax_spec.grid(True, alpha=0.12, color="#ffffff")

        self._line_amp, = self.ax_amp.plot([], [], color="#4b5563", linewidth=1.8)
        self._line_spec, = self.ax_spec.plot([], [], color="#f59e0b", linewidth=1.2, label="原始频谱")
        self._fill_spec = None
        self._line_conv_spec, = self.ax_spec.plot([], [], color="#06b6d4", linewidth=1.0, label="转换后")
        self._fill_conv = None
        self.ax_spec.legend(loc="upper right", fontsize=8, labelcolor="#6b7280",
                            facecolor="#1a1a2e", edgecolor="#374151", framealpha=0.8)

        guide = (
            "橙色=原始频谱  青色=转换后频谱  |  "
            "横轴=频率(Hz) 对数刻度  纵轴=能量(dB)  |  "
            "尖峰=谐波  谐波间距=基频  男声谐波间距窄  女声谐波间距宽  |  "
            "男声: 基频~85-180Hz 高频弱   女声: 基频~165-255Hz 高频强"
        )
        self.canvas.figure.text(0.5, 0.01, guide, ha="center", va="bottom",
                                fontsize=7.5, color="#6b7280", style="italic",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f5", edgecolor="#dde1e7", alpha=0.9))
        self.canvas.figure.tight_layout(rect=[0, 0.08, 1, 1])
        self.canvas.draw_idle()

    def _update_amp_chart(self, amplitude: float):
        self._amp_history.append(amplitude)
        if len(self._amp_history) > self.MAX_CHART_POINTS:
            self._amp_history.pop(0)
        xs = list(range(len(self._amp_history)))
        self._line_amp.set_data(xs, self._amp_history)
        if self._amp_history:
            amp_max = max(max(self._amp_history) * 1.3, 0.01)
            self.ax_amp.set_ylim(0, amp_max)
        self.ax_amp.set_xlim(-0.5, self.MAX_CHART_POINTS - 1 + 0.5)
        self.canvas.draw_idle()

    def _on_preview_toggle(self, target: str, checked: bool):
        if checked:
            self._preview_target = target
            # 互斥: 关闭另一个按钮
            if target == "female":
                self.preview_male_btn.setChecked(False)
                self.ax_spec.set_title("实时频谱 — 预览女声转换", fontsize=12, fontweight="bold", color="#ec4899")
            else:
                self.preview_female_btn.setChecked(False)
                self.ax_spec.set_title("实时频谱 — 预览男声转换", fontsize=12, fontweight="bold", color="#3b82f6")
        else:
            self._preview_target = None
            self._line_conv_spec.set_data([], [])
            if self._fill_conv is not None:
                self._fill_conv.remove()
                self._fill_conv = None
            self.ax_spec.set_title("实时频谱 (Frequency Spectrum)", fontsize=12, fontweight="bold", color="#374151")
            self.canvas.draw_idle()

    def _update_spectrum(self, audio: np.ndarray):
        n_fft = 8192
        if audio.size < n_fft:
            return
        y = audio.astype(np.float64)
        frame = y[-n_fft:]
        window = np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame * window))
        # ref=np.max(spec): correct frequency-domain reference
        spec_db = librosa.amplitude_to_db(spec, ref=np.max, top_db=60)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.SAMPLE_RATE)
        spec_db = np.convolve(spec_db, np.ones(5)/5.0, mode='same')
        # noise floor: EMA-tracked → stable flat baseline
        current_floor = float(np.percentile(spec_db, 10))
        self._noise_floor = 0.85 * self._noise_floor + 0.15 * current_floor
        spec_db = np.maximum(spec_db, self._noise_floor)
        mask = (freqs >= 50) & (freqs <= 5000)
        x, y = freqs[mask], spec_db[mask]
        self._line_spec.set_data(x, y)
        if self._fill_spec is not None:
            self._fill_spec.remove()
        self._fill_spec = self.ax_spec.fill_between(x, -40, y, color="#f59e0b", alpha=0.35, linewidth=0)
        self.canvas.draw_idle()

    def _preview_convert(self, audio: np.ndarray):
        """实时预览转换: 对短音频块运行 WORLD 管线, 返回转换后的频谱"""
        try:
            y = audio.astype(np.float64)
            sr = self.SAMPLE_RATE
            target = self._preview_target
            strength = self.strength_slider.value() / 100.0

            f0, t = pw.harvest(y, sr, f0_floor=50.0, f0_ceil=500.0)
            f0 = pw.stonemask(y, f0, t, sr)
            sp = pw.cheaptrick(y, f0, t, sr)
            ap = pw.d4c(y, f0, t, sr)

            voiced_f0 = f0[f0 > 0]
            mean_f0 = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
            strength = _clamp_control(strength)
            brightness = 0.35 + strength * 0.45
            f0_ratio = _compute_f0_ratio(mean_f0, target, strength)
            formant_factor = _compute_formant_factor(target, f0_ratio, strength)

            f0_converted = _smooth_f0(f0 * f0_ratio)
            sp_converted = _warp_spectral_envelope(sp, sr, formant_factor)
            sp_converted = _apply_spectral_tilt(sp_converted, target, brightness)
            sp_converted = _mix_with_original(sp, sp_converted, target, strength)
            sp_converted = _smooth_spectral_envelope(sp_converted, window=3)
            ap_converted = _adjust_aperiodicity(ap, f0_converted, f0_ratio, target)
            y_conv = pw.synthesize(f0_converted, sp_converted, ap_converted, sr)

            if y_conv.size:
                peak = float(np.max(np.abs(y_conv)))
                if peak > 1.0:
                    y_conv = y_conv / peak

            n_fft = 4096
            conv_audio = y_conv.astype(np.float64)
            if conv_audio.size < n_fft:
                self._line_conv_spec.set_data([], [])
                if self._fill_conv is not None:
                    self._fill_conv.remove()
                    self._fill_conv = None
                return
            frame = conv_audio[-n_fft:]
            window = np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(frame * window))
            spec_conv_db = librosa.amplitude_to_db(spec, ref=np.max, top_db=60)
            spec_conv_db = np.convolve(spec_conv_db, np.ones(7)/7.0, mode='same')
            current_floor = float(np.percentile(spec_conv_db, 10))
            self._noise_floor_conv = 0.85 * self._noise_floor_conv + 0.15 * current_floor
            spec_conv_db = np.maximum(spec_conv_db, self._noise_floor_conv)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / self.SAMPLE_RATE)
            mask = (freqs >= 50) & (freqs <= 5000)
            xc, yc = freqs[mask], spec_conv_db[mask]
            self._line_conv_spec.set_data(xc, yc)
            if self._fill_conv is not None:
                self._fill_conv.remove()
            self._fill_conv = self.ax_spec.fill_between(xc, -40, yc, color="#06b6d4", alpha=0.30, linewidth=0)
        except Exception:
            self._line_conv_spec.set_data([], [])
            if self._fill_conv is not None:
                self._fill_conv.remove()
                self._fill_conv = None

    # ── 样式表 ───────────────────────────────────────────────────────
    def _setup_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #eceff4;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #dde1e7;
                border-radius: 8px;
                margin-top: 16px;
                padding: 18px 12px 10px 12px;
                font-size: 13px;
                font-weight: bold;
                color: #475569;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #475569;
            }
            QPushButton {
                background-color: #4b5563;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #374151;
            }
            QPushButton:pressed {
                background-color: #1f2937;
            }
            QPushButton:disabled {
                background-color: #e5e7eb;
                color: #9ca3af;
            }
            QPushButton#export_btn, QPushButton#upload_btn {
                background-color: #ffffff;
                color: #4b5563;
                border: 1px solid #c4cad4;
            }
            QPushButton#export_btn:hover, QPushButton#upload_btn:hover {
                background-color: #f3f4f6;
                border-color: #9ca3af;
            }
            QPushButton#export_btn:disabled, QPushButton#upload_btn:disabled {
                background-color: #f9fafb;
                color: #d1d5db;
                border-color: #e5e7eb;
            }
            QPushButton#to_female_btn {
                background-color: #a55164;
            }
            QPushButton#to_female_btn:hover {
                background-color: #8b3f51;
            }
            QPushButton#to_male_btn {
                background-color: #4f697f;
            }
            QPushButton#to_male_btn:hover {
                background-color: #3d5367;
            }
            QLabel {
                color: #374151;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #dde1e7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                margin: -6px 0;
                background: #4b5563;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #374151;
            }
            QPlainTextEdit {
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 5px;
                font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 12px;
                padding: 6px;
            }
            QCheckBox {
                font-size: 12px;
                color: #6b7280;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border-radius: 3px;
                border: 1.5px solid #c4cad4;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #4b5563;
                border-color: #4b5563;
            }
            QStatusBar {
                background: #ffffff;
                border-top: 1px solid #e5e7eb;
                color: #6b7280;
            }
        """)
        self.start_btn.setObjectName("start_btn")
        self.stop_btn.setObjectName("stop_btn")
        self.export_btn.setObjectName("export_btn")
        self.upload_btn.setObjectName("upload_btn")
        self.to_female_btn.setObjectName("to_female_btn")
        self.to_male_btn.setObjectName("to_male_btn")

    # ── 按钮状态管理 ─────────────────────────────────────────────────
    def _update_button_states(self, state: str):
        has_data = self._full_audio_for_export is not None and self._full_audio_for_export.size > 0
        warm = self._warmed_up
        if state == "idle":
            self.start_btn.setEnabled(warm)
            self.stop_btn.setEnabled(False)
            self.export_btn.setEnabled(has_data)
            self.to_female_btn.setEnabled(has_data)
            self.to_male_btn.setEnabled(has_data)
            self.upload_btn.setEnabled(True)
        elif state == "recording":
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.export_btn.setEnabled(False)
            self.to_female_btn.setEnabled(False)
            self.to_male_btn.setEnabled(False)
            self.upload_btn.setEnabled(False)
        elif state == "stopped":
            self.start_btn.setEnabled(warm)
            self.stop_btn.setEnabled(False)
            self.export_btn.setEnabled(has_data)
            self.to_female_btn.setEnabled(has_data)
            self.to_male_btn.setEnabled(has_data)
            self.upload_btn.setEnabled(True)

    # ── 模型预热 ─────────────────────────────────────────────────────
    def _warmup_model(self):
        try:
            _load_model()
            # 预热 librosa.yin 的 Numba JIT 编译，避免首次预测等待 5-10s
            _warmup = np.random.randn(22050).astype(np.float32)
            librosa.yin(_warmup, fmin=50, fmax=500, sr=22050)
            self._warmup_msg = "模型加载完成 - 就绪"
        except Exception as e:
            self._warmup_msg = f"模型预热失败，将在首次预测时重试: {e}"
        finally:
            self._warmed_up = True
            self._warmup_done.emit()

    def _on_warmup_done(self):
        self.start_btn.setText("开始采集")
        self.statusBar().showMessage(self._warmup_msg, 5000)
        self._update_button_states("idle")

    # ── 音频回调 ─────────────────────────────────────────────────────
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        if self._ring is not None:
            chunk = indata[:, 0].copy()
            self._ring.write(chunk)
            self._full_audio_chunks.append(chunk)

    # ── 开始采集 ─────────────────────────────────────────────────────
    def _on_start(self):
        try:
            self._ring = AudioRingBuffer(capacity_seconds=300.0, sr=self.SAMPLE_RATE)
            self._full_audio_for_export = None
            self._full_audio_chunks = []
            self._first_prediction_done = False
            self._unknown_streak = 0
            self._last_displayed_gender = "?"
            self._gender_streak = 0
            self._pending_gender = None

            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=2048,
                callback=self._audio_callback,
            )
            self._stream.start()

            self._amp_history.clear()
            self._noise_floor = -40.0
            self._noise_floor_conv = -40.0
            self._clear_charts()

            self._chart_timer = QTimer(self)
            self._chart_timer.timeout.connect(self._tick_chart)
            self._chart_timer.start(self.CHART_INTERVAL_MS)

            self._predict_timer = QTimer(self)
            self._predict_timer.timeout.connect(self._tick_predict)
            self._predict_timer.start(self.PREDICT_INTERVAL_MS)

            self._update_button_states("recording")
            self.rec_status.setText("采集中...")
            self.statusBar().showMessage("开始采集")
        except sd.PortAudioError as e:
            QMessageBox.warning(self, "麦克风错误", f"无法打开麦克风:\n{e}")

    # ── 停止采集 ─────────────────────────────────────────────────────
    def _on_stop(self):
        if self._chart_timer:
            self._chart_timer.stop()
            self._chart_timer = None

        if self._predict_timer:
            self._predict_timer.stop()
            self._predict_timer = None

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._full_audio_chunks:
            self._full_audio_for_export = np.concatenate(self._full_audio_chunks)
        elif self._ring:
            self._full_audio_for_export = self._ring.get_full()
        if self._ring:
            self._ring = None

        self._preview_target = None
        self.preview_female_btn.setChecked(False)
        self.preview_male_btn.setChecked(False)
        self.ax_spec.set_title("实时频谱 (Frequency Spectrum)", fontsize=12, fontweight="bold", color="#374151")

        self._update_button_states("stopped")
        dur = float(self._full_audio_for_export.size / self.SAMPLE_RATE) if self._full_audio_for_export is not None else 0
        self.rec_status.setText(f"已停止 — 录制 {dur:.1f}s")
        self.statusBar().showMessage(f"采集停止，共 {dur:.1f} 秒")

    # ── 高速图表刷新（每 100ms 直接从缓冲区取幅值+频率）─────────────
    def _tick_chart(self):
        if self._ring is None:
            return
        dur = self._ring.total_samples / self.SAMPLE_RATE
        if not self._first_prediction_done:
            self.rec_status.setText(f"采集中... {dur:.0f}s | 预测准备中...")
        else:
            self.rec_status.setText(f"采集中... {dur:.0f}s")
        chunk = self._ring.get_last(0.3)
        if chunk is None or chunk.size == 0:
            return

        amplitude = float(np.max(np.abs(chunk)))

        self._update_amp_chart(amplitude)

        spec_audio = self._ring.get_last(0.4)
        if spec_audio is not None and spec_audio.size >= 256:
            self._update_spectrum(spec_audio)
            # 实时预览转换(每300ms运行一次, 避免阻塞)
            if self._preview_target is not None:
                self._preview_tick += 1
                if self._preview_tick % 3 == 0:
                    self._preview_convert(spec_audio)

    # ── 定时预测 ─────────────────────────────────────────────────────
    def _tick_predict(self):
        if self._ring is None:
            return
        if self._predict_worker is not None and self._predict_worker.isRunning():
            return  # 上一轮还在跑，跳过
        audio = self._ring.get_last(self.WINDOW_SECONDS)
        if audio is None:
            return
        self._predict_worker = PredictWorker(audio, sr=self.SAMPLE_RATE, enable_vad=self.vad_checkbox.isChecked())
        self._predict_worker.result_ready.connect(self._on_prediction_result)
        self._predict_worker.error_occurred.connect(self._on_prediction_error)
        self._predict_worker.finished.connect(lambda: setattr(self, "_predict_worker", None))
        self._predict_worker.start()

    def _on_prediction_result(self, result: dict):
        self._first_prediction_done = True
        gender = result.get("gender", "unknown")
        if gender == "unknown":
            self._unknown_streak += 1
            self._gender_streak = 0
            self._pending_gender = None
            if self._unknown_streak >= 2:
                if self._last_displayed_gender != "unknown":
                    self.gender_label.setText("未检测到人声")
                    self.confidence_label.setText("置信度：--%")
                    self._last_displayed_gender = "unknown"
        else:
            self._unknown_streak = 0
            new_label = str(gender)
            if self._last_displayed_gender == new_label:
                self._gender_streak = 0
                self._pending_gender = None
            else:
                if self._pending_gender == new_label:
                    self._gender_streak += 1
                else:
                    self._pending_gender = new_label
                    self._gender_streak = 1
                if self._gender_streak >= 3:
                    self.gender_label.setText(gender)
                    self.confidence_label.setText(f"置信度：{result['confidence']}%")
                    self._last_displayed_gender = new_label
                    self._gender_streak = 0
                    self._pending_gender = None

        # 调试信息
        dbg = result.get("debug", {})
        lines = []
        if "message" in dbg:
            lines.append(dbg["message"])
        else:
            lines.append(f"预测: {dbg.get('raw_prediction', '?')} | 概率: {dbg.get('model_proba', '?')}")
            lines.append(f"时长: {dbg.get('duration', '?')}")
            feats = dbg.get("features", {})
            if feats:
                keys = ["meanfreq", "meanfun", "IQR", "sd", "sp.ent", "sfm",
                        "meandom", "mindom", "maxdom", "modindx", "skew", "kurt"]
                line = "特征: " + " ".join(f"{k}={feats[k]}" for k in keys if k in feats)
                lines.append(line)
            aux = dbg.get("aux_features", {})
            if aux:
                lines.append("辅助: " + " ".join(f"{k}={v}" for k, v in aux.items()))
            outliers = dbg.get("outliers", [])
            if outliers:
                lines.append("[异常] " + ", ".join(outliers))
            else:
                lines.append("[正常] 所有特征在训练数据范围内")
        self.debug_box.setPlainText("\n".join(lines))

    def _on_prediction_error(self, error: str):
        self.statusBar().showMessage(f"预测错误: {error}")
        self.debug_box.setPlainText(f"预测失败: {error}")

    # ── 导出录音 ─────────────────────────────────────────────────────
    def _on_export(self):
        if self._full_audio_for_export is None or self._full_audio_for_export.size == 0:
            return
        rec_dir = str(Path(__file__).resolve().parent.parent / "recordings")
        Path(rec_dir).mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "导出录音", f"{rec_dir}/recording.wav", "WAV 文件 (*.wav)"
        )
        if not path:
            return
        try:
            sf.write(path, self._full_audio_for_export, self.SAMPLE_RATE)
            self.statusBar().showMessage(f"已导出: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    # ── 声音转换 ─────────────────────────────────────────────────────
    def _on_convert(self, target: str):
        if self._full_audio_for_export is None or self._full_audio_for_export.size == 0:
            return
        default_name = "converted_female.wav" if target == "female" else "converted_male.wav"
        rec_dir = str(Path(__file__).resolve().parent.parent / "recordings")
        Path(rec_dir).mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "保存转换结果", f"{rec_dir}/{default_name}", "WAV 文件 (*.wav)"
        )
        if not path:
            return

        strength = float(self.strength_slider.value()) / 100.0
        self.statusBar().showMessage("正在转换，请稍候...")
        self._convert_worker = ConvertWorker(
            self._full_audio_for_export, self.SAMPLE_RATE, target, strength, path
        )
        self._convert_worker.finished.connect(self._on_convert_finished)
        self._convert_worker.error_occurred.connect(self._on_convert_error)
        self._convert_worker.start()

    def _on_convert_finished(self, path: str):
        self.statusBar().showMessage(f"转换完成: {path}")
        QMessageBox.information(self, "转换完成", f"已保存到:\n{path}")

    def _on_convert_error(self, error: str):
        QMessageBox.critical(self, "转换失败", error)

    # ── 女声强度滑块 ─────────────────────────────────────────────────
    def _on_strength_changed(self, value: int):
        self.strength_value_label.setText(f"{value}%")

    # ── 文件上传测试 ─────────────────────────────────────────────────
    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "音频文件 (*.wav *.mp3 *.ogg *.flac);;所有文件 (*)"
        )
        if not path:
            return
        self.upload_status.setText("正在分析...")
        self.statusBar().showMessage("正在分析音频文件...")

        try:
            y, sr = librosa.load(path, sr=self.SAMPLE_RATE, mono=True)
        except Exception as e:
            self.upload_status.setText(f"读取失败: {e}")
            return

        self._upload_worker = PredictWorker(y, sr=self.SAMPLE_RATE, enable_vad=self.vad_checkbox.isChecked())
        self._upload_worker.result_ready.connect(self._on_upload_result)
        self._upload_worker.error_occurred.connect(lambda e: self.upload_status.setText(f"分析失败: {e}"))
        self._upload_worker.finished.connect(lambda: setattr(self, "_upload_worker", None))
        self._upload_worker.start()

    def _on_upload_result(self, result: dict):
        gender = result.get("gender", "unknown")
        if gender == "unknown":
            self.gender_label.setText("未检测到人声")
            self.confidence_label.setText("置信度：--%")
        else:
            self.gender_label.setText(gender)
            self.confidence_label.setText(f"置信度：{result['confidence']}%")

        # 显示完整调试信息
        dbg = result.get("debug", {})
        lines = []
        if "message" in dbg:
            lines.append(dbg["message"])
        else:
            lines.append(f"原始预测: {dbg.get('raw_prediction', '?')} | 概率: {dbg.get('model_proba', '?')}")
            lines.append(f"时长: {dbg.get('duration', '?')}")
            lines.append("─" * 40)
            lines.append("全部特征:")
            feats = dbg.get("features", {})
            for k, v in feats.items():
                marker = ""
                if k in TRAINING_REF and isinstance(v, float):
                    _, _, rmin, rmax = TRAINING_REF[k]
                    if v < rmin * 0.5 or v > rmax * 1.5:
                        marker = f"  ← 偏离范围 [{rmin:.4f}, {rmax:.4f}]"
                lines.append(f"  {k}: {v}{marker}")
            outliers = dbg.get("outliers", [])
            if outliers:
                lines.insert(2, "─" * 40)
                lines.insert(2, "[异常特征] " + ", ".join(outliers))
            else:
                lines.insert(2, "[正常] 所有特征在训练数据范围内")
        self.debug_box.setPlainText("\n".join(lines))
        self.upload_status.setText(f"分析完成: {gender}")
        self.statusBar().showMessage(f"文件分析完成: {gender}")

    # ── 图表清理 ─────────────────────────────────────────────────────
    def _clear_charts(self):
        self._line_amp.set_data([], [])
        self._line_spec.set_data([], [])
        self._line_conv_spec.set_data([], [])
        if self._fill_spec is not None:
            self._fill_spec.remove()
            self._fill_spec = None
        if self._fill_conv is not None:
            self._fill_conv.remove()
            self._fill_conv = None
        self.ax_amp.set_ylim(0, 0.01)
        self.canvas.draw_idle()

    # ── 窗口关闭 ─────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self._chart_timer:
            self._chart_timer.stop()
        if self._predict_timer:
            self._predict_timer.stop()
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._predict_worker and self._predict_worker.isRunning():
            self._predict_worker.quit()
            self._predict_worker.wait(2000)
        if self._convert_worker and self._convert_worker.isRunning():
            self._convert_worker.quit()
            self._convert_worker.wait(2000)
        event.accept()


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VoiceGenderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
