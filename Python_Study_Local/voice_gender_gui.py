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
    _detect_voice_activity,
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

            # 有效语音比例检测（防止刚开口时大部分是静音导致误判）
            frame_len = int(self._sr * 0.1)
            if frame_len >= 64 and y.size >= frame_len:
                n_frames = y.size // frame_len
                frame_rms = np.array([
                    float(np.sqrt(np.mean(y[i * frame_len:(i + 1) * frame_len] ** 2)))
                    for i in range(n_frames)
                ])
                active_ratio = float(np.mean(frame_rms > 0.004))
            else:
                active_ratio = 1.0 if float(np.sqrt(np.mean(y ** 2))) > 0.004 else 0.0

            if active_ratio < 0.15:
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
                    "debug": {"message": "有效语音不足"},
                })
                return
            if self._enable_vad and not _detect_voice_activity(y, self._sr):
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
                    "debug": {"message": "未检测到人声"},
                })
                return

            model, feature_names, label_mapping = _load_model()
            y_for_feature = _ensure_min_duration(y, self._sr, min_duration=1.2)
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

            f0_converted = _smooth_f0(f0 * f0_ratio, window=5)
            sp_converted = _warp_spectral_envelope(sp, sr, formant_factor)
            sp_converted = _apply_spectral_tilt(sp_converted, target, brightness)
            sp_converted = _mix_with_original(sp, sp_converted, target, strength)
            sp_converted = _smooth_spectral_envelope(sp_converted, window=3)
            ap_converted = _adjust_aperiodicity(ap, f0_ratio, target)
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
    MAX_CHART_POINTS = 30
    PREDICT_INTERVAL_MS = 2000
    WINDOW_SECONDS = 5.0
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
        self._predict_worker: PredictWorker | None = None
        self._convert_worker: ConvertWorker | None = None
        self._upload_worker: PredictWorker | None = None
        self._warmed_up = False
        self._ui_ready = False
        self._full_audio_for_export: np.ndarray | None = None
        self._amp_history: list[float] = []
        self._freq_history: list[float] = []

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
        left_layout.setSpacing(10)

        # 录制控制
        rec_group = QGroupBox("录制控制")
        rec_layout = QVBoxLayout(rec_group)
        rec_layout.setSpacing(10)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.start_btn = QPushButton("开始采集")
        self.start_btn.setMinimumHeight(34)
        self.stop_btn = QPushButton("停止采集")
        self.stop_btn.setMinimumHeight(34)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        rec_layout.addLayout(btn_row)
        self.export_btn = QPushButton("导出录音 WAV")
        self.export_btn.setMinimumHeight(30)
        rec_layout.addWidget(self.export_btn)
        self.rec_status = QLabel("就绪")
        self.rec_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

        self.ax_freq = self.canvas.figure.add_subplot(2, 1, 2)
        self.ax_freq.set_facecolor("#ffffff")
        self.ax_freq.set_title("归一化频率", fontsize=12, fontweight="bold", color="#374151")
        self.ax_freq.set_ylabel("归一化频率", fontsize=10, color="#6b7280")
        self.ax_freq.set_xlabel("采样点", fontsize=10, color="#6b7280")
        self.ax_freq.grid(True, alpha=0.2, color="#d1d5db")
        self.ax_freq.set_xlim(0, self.MAX_CHART_POINTS)
        self.ax_freq.set_ylim(0.03, 0.28)
        self.ax_freq.tick_params(labelsize=9, colors="#6b7280")

        self._line_amp, = self.ax_amp.plot([], [], color="#4b5563", linewidth=1.8)
        self._line_freq, = self.ax_freq.plot([], [], color="#5b7f95", linewidth=1.8)
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()

    def _update_charts(self, amplitude: float, frequency: float):
        self._amp_history.append(amplitude)
        self._freq_history.append(frequency)
        if len(self._amp_history) > self.MAX_CHART_POINTS:
            self._amp_history.pop(0)
            self._freq_history.pop(0)

        xs = list(range(len(self._amp_history)))
        self._line_amp.set_data(xs, self._amp_history)
        self._line_freq.set_data(xs, self._freq_history)

        # 动态调整幅度轴
        if self._amp_history:
            amp_max = max(max(self._amp_history) * 1.3, 0.01)
            self.ax_amp.set_ylim(0, amp_max)

        self.ax_amp.set_xlim(-0.5, max(self.MAX_CHART_POINTS - 1, len(xs) - 1) + 0.5)
        self.ax_freq.set_xlim(-0.5, max(self.MAX_CHART_POINTS - 1, len(xs) - 1) + 0.5)
        self.canvas.draw_idle()

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
            self._ring.write(indata[:, 0].copy())

    # ── 开始采集 ─────────────────────────────────────────────────────
    def _on_start(self):
        try:
            self._ring = AudioRingBuffer(capacity_seconds=15.0, sr=self.SAMPLE_RATE)
            self._full_audio_for_export = None

            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=2048,
                callback=self._audio_callback,
            )
            self._stream.start()

            self._amp_history.clear()
            self._freq_history.clear()
            self._clear_charts()

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
        if self._predict_timer:
            self._predict_timer.stop()
            self._predict_timer = None

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._ring:
            self._full_audio_for_export = self._ring.get_full()
            self._ring = None

        self._update_button_states("stopped")
        dur = float(self._full_audio_for_export.size / self.SAMPLE_RATE) if self._full_audio_for_export is not None else 0
        self.rec_status.setText(f"已停止 — 录制 {dur:.1f}s")
        self.statusBar().showMessage(f"采集停止，共 {dur:.1f} 秒")

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
        gender = result.get("gender", "unknown")
        if gender == "unknown":
            self.gender_label.setText("未检测到人声")
            self.confidence_label.setText("置信度：--%")
        else:
            self.gender_label.setText(gender)
            self.confidence_label.setText(f"置信度：{result['confidence']}%")

        df = result.get("display_features", {})
        self._update_charts(df.get("amplitude", 0.0), df.get("mean_frequency", 0.0))

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
        path, _ = QFileDialog.getSaveFileName(
            self, "导出录音", "recording.wav", "WAV 文件 (*.wav)"
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
        path, _ = QFileDialog.getSaveFileName(
            self, "保存转换结果", default_name, "WAV 文件 (*.wav)"
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
        self._line_freq.set_data([], [])
        self.ax_amp.set_ylim(0, 0.01)
        self.canvas.draw_idle()

    # ── 窗口关闭 ─────────────────────────────────────────────────────
    def closeEvent(self, event):
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
