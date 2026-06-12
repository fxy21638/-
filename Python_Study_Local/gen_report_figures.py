"""生成信号与系统项目报告所需的全部图表"""
import os, sys, time
from pathlib import Path

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import xgboost as xgb
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from main import (
    extract_audio_features, _ensure_min_duration, _load_model,
    _smooth_f0, _smooth_spectral_envelope, _warp_spectral_envelope,
    _apply_spectral_tilt, _mix_with_original, _compute_f0_ratio,
    _compute_formant_factor, _adjust_aperiodicity,
)

SR = 22050
DATA = Path(r"D:\new_document\信号与系统\信号与系统\data")
MODEL_DIR = Path(r"D:\new_document\Document\voice")
OUT_DIR = Path(r"d:\new_document\Python_Study_Local\report_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model, feature_names, label_mapping = _load_model()

# ═══════════════════════════════════════════════════════════════════════════
# 图1: 男声 vs 女声波形 + 语谱图对比
# ═══════════════════════════════════════════════════════════════════════════
def fig1_spectrograms():
    male_fp = sorted((DATA / "male").glob("*.wav"))[100]
    female_fp = sorted((DATA / "female").glob("*.wav"))[100]

    fig, axes = plt.subplots(2, 2, figsize=(16, 8),
                              gridspec_kw={"height_ratios": [1, 2.5], "width_ratios": [1, 1]})

    for col, (fp, label, color) in enumerate([(male_fp, "男性 (Male)", "#3b82f6"),
                                                (female_fp, "女性 (Female)", "#ec4899")]):
        y, sr_orig = sf.read(str(fp), dtype="float32")
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr_orig != SR: y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)

        # 上排: 波形
        ax_wave = axes[0, col]
        t = np.arange(len(y)) / SR
        ax_wave.plot(t, y, color=color, linewidth=0.6)
        ax_wave.set_title(f"{label} — 时域波形", fontsize=13, fontweight="bold", color=color)
        ax_wave.set_ylabel("振幅", fontsize=10)
        ax_wave.set_xlim(0, t[-1])
        ax_wave.set_ylim(-0.4, 0.4)
        ax_wave.grid(True, alpha=0.2)

        # 下排: 语谱图
        ax_spec = axes[1, col]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048)), ref=np.max)
        librosa.display.specshow(D, sr=SR, hop_length=512, x_axis="time",
                                  y_axis="log", ax=ax_spec, cmap="magma")
        ax_spec.set_title(f"{label} — 语谱图 (Spectrogram)", fontsize=13, fontweight="bold", color=color)
        ax_spec.set_ylabel("频率 (Hz)", fontsize=10)
        ax_spec.set_xlabel("时间 (s)", fontsize=10)
        ax_spec.set_ylim(50, 5000)

    fig.suptitle("图1: 男/女声波形与语谱图对比", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_spectrograms.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 图1: 语谱图对比")

# ═══════════════════════════════════════════════════════════════════════════
# 图2: F0 (基频) 轮廓对比 — 上下分栏，浊音点 + 插值线
# ═══════════════════════════════════════════════════════════════════════════
def fig2_f0_contour():
    # 选F0差异明显的典型样本: 男~99Hz, 女~258Hz
    male_fp = sorted((DATA / "male").glob("*.wav"))[150]
    female_fp = sorted((DATA / "female").glob("*.wav"))[20]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"hspace": 0.18})

    configs = [
        (ax1, male_fp, "男性 (Male)", "#2563eb", (60, 200), "男声典型范围\n~85–180 Hz"),
        (ax2, female_fp, "女性 (Female)", "#db2777", (140, 400), "女声典型范围\n~165–255 Hz"),
    ]

    for ax, fp, label, color, band, band_label in configs:
        y, sr_orig = sf.read(str(fp), dtype="float32")
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr_orig != SR: y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)

        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=SR, fill_na=None)
        times = librosa.times_like(f0, sr=SR)

        # 分离浊音/清音
        valid = ~np.isnan(f0)
        voiced_t = times[valid]
        voiced_f = f0[valid]
        unvoiced_t = times[~valid]
        mean_f0 = np.mean(voiced_f)

        # 浊音F0 — 加粗的实线（只画浊音段）
        ax.plot(voiced_t, voiced_f, color=color, linewidth=1.8, alpha=0.9)

        # 清音段用很淡的灰色线标记在底部
        if len(unvoiced_t) > 0:
            # 找连续的清音区间
            gaps = np.diff(np.concatenate([[False], ~valid, [False]]).astype(int))
            gap_starts = np.where(gaps == 1)[0]
            gap_ends = np.where(gaps == -1)[0]
            for gs, ge in zip(gap_starts, gap_ends):
                if gs < len(times) and ge <= len(times):
                    ax.axvspan(times[gs], times[min(ge, len(times)) - 1],
                               alpha=0.08, color="#9ca3af", zorder=0)

        # 平均F0 — 醒目的虚线 + 数值标记
        ax.axhline(y=mean_f0, color="#dc2626", linestyle="--", linewidth=1.8, alpha=0.85)
        ax.text(times[-1] * 0.995, mean_f0 + 12, f"均值 {mean_f0:.0f} Hz",
                ha="right", va="bottom", fontsize=10.5, fontweight="bold",
                color="#dc2626", bbox=dict(boxstyle="round,pad=0.25", fc="white",
                alpha=0.85, ec="#fca5a5"))

        # 典型F0范围带
        ax.axhspan(band[0], band[1], alpha=0.08, color=color)
        ax.text(0.01, band[1] - 6, band_label, ha="left", va="top",
                fontsize=9, color=color, alpha=0.7, fontstyle="italic")

        ax.set_ylabel("基频 F0 (Hz)", fontsize=11)
        ax.set_ylim(30, 520)
        ax.set_xlim(0, times[-1])
        ax.grid(True, alpha=0.12)
        # 加文字标签在左上角
        ax.text(0.01, 0.94, label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=color, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec=color))

    ax2.set_xlabel("时间 (s)", fontsize=11)
    fig.suptitle("图2: 男/女声基频 (F0) 轮廓对比", fontsize=15, fontweight="bold", y=1.01)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.07)
    fig.savefig(OUT_DIR / "fig2_f0_contour.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 图2: F0轮廓对比")

# ═══════════════════════════════════════════════════════════════════════════
# 图3: 特征重要性排名
# ═══════════════════════════════════════════════════════════════════════════
def fig3_feature_importance():
    importances = model.feature_importances_
    idx = np.argsort(importances)
    top_n = 15
    top_idx = idx[-top_n:]
    names_top = [feature_names[i] for i in top_idx]
    values_top = importances[top_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#3b82f6" if v > 0.05 else "#93c5fd" for v in values_top]
    bars = ax.barh(range(top_n), values_top, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_top, fontsize=10)
    ax.set_xlabel("特征重要性 (Feature Importance)", fontsize=11)
    ax.set_title("图3: XGBoost 模型特征重要性排名 (Top 15)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, values_top.max() * 1.15)
    ax.grid(True, alpha=0.15, axis="x")

    for bar, val in zip(bars, values_top):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#374151")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 图3: 特征重要性")

# ═══════════════════════════════════════════════════════════════════════════
# 图4: 频谱对比 (FFT 幅度谱)
# ═══════════════════════════════════════════════════════════════════════════
def fig4_spectrum():
    male_fp = sorted((DATA / "male").glob("*.wav"))[100]
    female_fp = sorted((DATA / "female").glob("*.wav"))[100]

    fig, ax = plt.subplots(figsize=(12, 5))

    for fp, label, color, ls in [(male_fp, "男性", "#3b82f6", "-"),
                                   (female_fp, "女性", "#ec4899", "-")]:
        y, sr_orig = sf.read(str(fp), dtype="float32")
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr_orig != SR: y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)

        # 取中间稳定段做 FFT
        mid_start = len(y) // 3
        segment = y[mid_start:mid_start + 16384].astype(np.float64)
        window = np.hanning(len(segment))
        spec = np.abs(np.fft.rfft(segment * window))
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)
        freqs = np.fft.rfftfreq(len(segment), 1.0 / SR)
        spec_db = np.convolve(spec_db, np.ones(7) / 7, mode="same")

        mask = (freqs >= 40) & (freqs <= 4000)
        ax.plot(freqs[mask], spec_db[mask], color=color, linewidth=1.3, linestyle=ls, label=label, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("频率 (Hz), 对数刻度", fontsize=11)
    ax.set_ylabel("幅度 (dB)", fontsize=11)
    ax.set_title("图4: 男/女声 FFT 幅度谱对比 (频域)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(50, 4000)
    ax.set_ylim(-55, 3)

    # 标注谐波
    for hz, label in [(120, "男声基频 ~120Hz"), (210, "女声基频 ~210Hz")]:
        ax.axvline(x=hz, color="gray" if "男" in label else "gray", linestyle=":", alpha=0.5, linewidth=0.8)
        # 不标注文字避免拥挤

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_spectrum.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 图4: 频谱对比")

# ═══════════════════════════════════════════════════════════════════════════
# 图5: 混淆矩阵
# ═══════════════════════════════════════════════════════════════════════════
def fig5_confusion_matrix():
    # 在全部数据集上做预测
    all_y_true = []
    all_y_pred = []

    for label_int, label_str in [(1, "male"), (0, "female")]:
        category_dir = DATA / label_str
        files = sorted(category_dir.glob("*.wav"))
        print(f"  预测 {label_str}/ ({len(files)} 文件)...")
        for i, fp in enumerate(files):
            try:
                y, sr_orig = sf.read(str(fp), dtype="float32")
                if y.ndim > 1: y = np.mean(y, axis=1)
                if sr_orig != SR: y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)
                y = _ensure_min_duration(y, SR, min_duration=1.2)
                df = extract_audio_features(y, SR, feature_names)
                pred = int(model.predict(df)[0])
                all_y_true.append(label_int)
                all_y_pred.append(pred)
            except Exception:
                pass
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{len(files)}")

    cm = confusion_matrix(all_y_true, all_y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["女性", "男性"])
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.set_title(f"图5: 混淆矩阵 (总准确率 = {acc:.1%})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图5: 混淆矩阵 (准确率={acc:.1%})")

# ═══════════════════════════════════════════════════════════════════════════
# 图6: VAD 人声检测示意图
# ═══════════════════════════════════════════════════════════════════════════
def fig6_vad_demo():
    # 用真实语音 + 人为插入静音间隙，清楚展示 VAD 工作原理
    male_fp = sorted((DATA / "male").glob("*.wav"))[30]
    y_raw, sr_orig = sf.read(str(male_fp), dtype="float32")
    if y_raw.ndim > 1: y_raw = np.mean(y_raw, axis=1)
    if sr_orig != SR: y_raw = librosa.resample(y_raw, orig_sr=sr_orig, target_sr=SR)

    # 取 3 秒并插入静音间隙
    y_full = y_raw[:int(3.5 * SR)].copy()
    y_full[int(1.2 * SR):int(1.7 * SR)] = 0    # 中间停顿
    y_full[int(2.8 * SR):] = 0                  # 尾部静音
    t_full = np.arange(len(y_full)) / SR

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                    gridspec_kw={"hspace": 0.12})

    # 上: 波形 + 静音区域高亮
    ax1.plot(t_full, y_full, color="#374151", linewidth=0.4)
    ax1.set_ylabel("振幅", fontsize=11)
    ax1.set_xlim(0, t_full[-1])
    ax1.set_ylim(-0.45, 0.45)
    ax1.grid(True, alpha=0.15)

    # 标记静音区间
    for start, end, label in [(0, 0.08, "句首"), (1.2, 1.7, "句中停顿"), (2.8, 3.5, "句尾")]:
        ax1.axvspan(start, end, alpha=0.18, color="#ef4444")
        ax1.text((start + end) / 2, 0.38, label, ha="center", va="center",
                 fontsize=9, color="#991b1b", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

    ax1.set_title("真实语音波形 (含静音段)", fontsize=12, fontweight="bold")

    # 下: 分帧 RMS + 自适应阈值
    frame_len = int(SR * 0.1)  # 100ms
    n_frames = len(y_full) // frame_len
    frame_rms = np.array([
        np.sqrt(np.mean(y_full[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])
    frame_times = (np.arange(n_frames) + 0.5) * frame_len / SR

    # 自适应阈值 (与 GUI 端 VAD 一致: 第20百分位 × 3)
    noise_floor = np.percentile(frame_rms, 20)
    vad_threshold = max(noise_floor * 3.0, 0.0015)
    active = frame_rms > vad_threshold
    active_ratio = np.mean(active)

    bar_colors = ["#22c55e" if a else "#ef4444" for a in active]
    ax2.bar(frame_times, frame_rms, width=0.09, color=bar_colors,
            alpha=0.78, edgecolor="white", linewidth=0.2)
    ax2.axhline(y=vad_threshold, color="#1e40af", linestyle="--", linewidth=1.8,
                label=f"自适应阈值 = {vad_threshold:.4f}  (第20百分位 × 3)")
    ax2.axhline(y=noise_floor, color="#9ca3af", linestyle=":", linewidth=1.2,
                label=f"噪声基底 = {noise_floor:.4f}  (第20百分位)")
    ax2.fill_between(frame_times, 0, vad_threshold, alpha=0.06, color="#1e40af")

    ax2.set_ylabel("RMS 能量", fontsize=11)
    ax2.set_xlabel("时间 (s)", fontsize=11)
    ax2.set_xlim(0, t_full[-1])
    ax2.legend(fontsize=9.5, loc="upper right", framealpha=0.85, edgecolor="#ddd")
    ax2.grid(True, alpha=0.15)

    # VAD 判定结果标注
    verdict = "有效语音 [是]" if active_ratio > 0.1 else "无效语音 [否]"
    verdict_color = "#166534" if active_ratio > 0.1 else "#991b1b"
    ax2.text(0.98, 0.94, f"有声帧占比: {active_ratio:.1%}  →  {verdict}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=11, fontweight="bold", color=verdict_color,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="#ddd"))

    fig.suptitle("图6: 人声活动检测 (VAD) — 自适应阈值 + 真实语音", fontsize=14, fontweight="bold", y=1.01)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08)
    fig.savefig(OUT_DIR / "fig6_vad_demo.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 图6: VAD演示")

# ═══════════════════════════════════════════════════════════════════════════
# 图7: 环形缓冲区原理
# ═══════════════════════════════════════════════════════════════════════════
def fig7_ringbuffer():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    cap = 20

    # 每个子图的数据: (title, data_array, write_pos, is_window_demo)
    panels = [
        ("t1: 写入前8个样本",
         [1.0, 1.0, 1.0, 1.0, 0.3, 0.8, 0.6, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         8, False),
        ("t2: 继续写入",
         [1.0, 1.0, 1.0, 1.0, 0.3, 0.8, 0.6, 0.5, 0.9, 0.7, 0.4, 0.2, 0.8, 0.6, 0.3, 0, 0, 0, 0, 0],
         16, False),
        ("t3: 提取滑动窗口(6样本)",
         [1.0, 1.0, 1.0, 1.0, 0.3, 0.8, 0.6, 0.5, 0.9, 0.7, 0.4, 0.2, 0.8, 0.6, 0.3, 0, 0, 0, 0, 0],
         -1, True),
        ("t4: 循环覆盖旧数据",
         [0.5, 0.3, 0.9, 1.0, 0.3, 0.8, 0.6, 0.5, 0.9, 0.7, 0.4, 0.2, 0.8, 0.6, 0.3, 0.4, 0.7, 0.2, 0.9, 0.1],
         3, False),
    ]

    for idx, (ax, (title, arr, wpos, is_window)) in enumerate(zip(axes, panels)):
        if is_window:
            colors_list = ["#d1d5db"] * cap
            for j in range(10, 16):
                colors_list[j] = "#fbbf24"
            ax.bar(range(cap), arr, color=colors_list, edgecolor="white", linewidth=0.5)
            ax.axvspan(9.5, 15.5, alpha=0.15, color="#f59e0b")
            ax.annotate("滑动窗口\n(最近6样本)", xy=(12.5, 0.85), fontsize=9, ha="center",
                        bbox=dict(boxstyle="round", facecolor="#fef3c7", alpha=0.9))
        else:
            idx_valid = [j for j, v in enumerate(arr) if v > 0]
            colors_list = ["#d1d5db"] * cap
            for j in idx_valid[-8:]:
                colors_list[j] = "#3b82f6"
            ax.bar(range(cap), arr, color=colors_list, edgecolor="white", linewidth=0.5)
            if 0 <= wpos < cap:
                ax.axvline(x=wpos - 0.5, color="#ef4444", linewidth=2, linestyle="--")
                ax.annotate(f"写指针\npos={wpos}", xy=(wpos - 0.5, 0.9), fontsize=8, color="#ef4444",
                            ha="center", fontweight="bold")
        ax.set_ylim(0, 1.2)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("缓冲区索引", fontsize=8)
        if idx == 0:
            ax.set_ylabel("样本值", fontsize=9)
        ax.set_xticks(range(0, cap, 4))
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("图7: 环形缓冲区 (Ring Buffer) 工作原理", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_ringbuffer.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 图7: 环形缓冲区")

# ═══════════════════════════════════════════════════════════════════════════
# 图8: 特征分布对比 (箱线图)
# ═══════════════════════════════════════════════════════════════════════════
def fig8_feature_boxplot():
    # 提取所有文件的关键特征做对比
    rows = []
    for label in ("male", "female"):
        files = sorted((DATA / label).glob("*.wav"))[:200]
        for fp in files:
            try:
                y, sr_orig = sf.read(str(fp), dtype="float32")
                if y.ndim > 1: y = np.mean(y, axis=1)
                if sr_orig != SR: y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)
                y = _ensure_min_duration(y, SR, min_duration=1.2)
                df = extract_audio_features(y, SR, feature_names)
                row = {"label": label}
                for fn in ["meanfreq", "meanfun", "IQR", "sp.ent", "sd", "skew", "kurt", "meandom"]:
                    row[fn] = float(df[fn].iloc[0])
                rows.append(row)
            except Exception:
                pass

    df_long = pd.DataFrame(rows)
    features_show = ["meanfreq", "meanfun", "IQR", "sp.ent", "sd", "skew", "kurt", "meandom"]
    labels_cn = ["meanfreq\n(频谱质心)", "meanfun\n(平均基频)", "IQR\n(四分位距)", "sp.ent\n(谱熵)",
                 "sd\n(质心标准差)", "skew\n(偏度)", "kurt\n(峰度)", "meandom\n(平均主频)"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (fn, cn) in enumerate(zip(features_show, labels_cn)):
        ax = axes[idx // 4, idx % 4]
        male_vals = df_long[df_long["label"] == "male"][fn].dropna()
        female_vals = df_long[df_long["label"] == "female"][fn].dropna()
        bp = ax.boxplot([male_vals, female_vals], labels=["男", "女"],
                         patch_artist=True, widths=0.5,
                         medianprops={"color": "black", "linewidth": 1.5})
        bp["boxes"][0].set_facecolor("#3b82f6")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#ec4899")
        bp["boxes"][1].set_alpha(0.6)
        ax.set_title(cn.replace("\n", " "), fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")
    fig.suptitle("图8: 男/女声关键特征分布对比 (箱线图)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig8_feature_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 图8: 特征箱线图")

# ═══════════════════════════════════════════════════════════════════════════
# 图9: WORLD 变声管线 (源-滤波器模型可视化)
# ═══════════════════════════════════════════════════════════════════════════
def fig9_world_pipeline():
    fp = sorted((DATA / "female").glob("*.wav"))[80]
    y, sr_orig = sf.read(str(fp), dtype="float32")
    if y.ndim > 1: y = np.mean(y, axis=1)
    if sr_orig != SR: y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)
    y = y.astype(np.float64)

    import pyworld as pw
    f0, t = pw.harvest(y, SR, f0_floor=50, f0_ceil=500)
    f0 = pw.stonemask(y, f0, t, SR)
    sp = pw.cheaptrick(y, f0, t, SR)
    ap = pw.d4c(y, f0, t, SR)

    # 转男声
    target = "male"
    strength = 0.6
    f0_ratio = _compute_f0_ratio(np.nanmean(f0), target, strength)
    formant_factor = _compute_formant_factor(target, strength)
    brightness = 0.35 + strength * 0.45

    f0_conv = _smooth_f0(f0 * f0_ratio, window=5)
    sp_warped = _warp_spectral_envelope(sp, SR, formant_factor)
    sp_tilt = _apply_spectral_tilt(sp_warped, target, brightness)
    sp_mix = _mix_with_original(sp, sp_tilt, target, strength)
    sp_smooth = _smooth_spectral_envelope(sp_mix, window=3)
    ap_adj = _adjust_aperiodicity(ap, f0, f0_ratio, target)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))

    # 原始 F0
    ax = axes[0, 0]
    ax.plot(t, f0, color="#ec4899", linewidth=1.2)
    ax.set_title("① 原始F0 (源激励)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Hz", fontsize=9)
    ax.set_ylim(0, 520)
    ax.grid(True, alpha=0.2)

    # 转换后 F0
    ax = axes[0, 1]
    ax.plot(t, f0_conv, color="#3b82f6", linewidth=1.2)
    ax.set_title(f"② F0缩放 ×{f0_ratio:.2f} → 男声", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 520)
    ax.grid(True, alpha=0.2)

    # F0 平滑前/后对比
    ax = axes[0, 2]
    ax.plot(t[:len(f0)], f0 * f0_ratio, color="gray", linewidth=0.6, alpha=0.5, label="缩放后(未平滑)")
    ax.plot(t[:len(f0_conv)], f0_conv, color="#3b82f6", linewidth=1.2, label="平滑后")
    ax.set_title("③ F0平滑 (中值+线性插值)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 520)
    ax.grid(True, alpha=0.2)

    # 原始频谱包络
    ax = axes[1, 0]
    spec_db = librosa.amplitude_to_db(sp.T, ref=np.max)
    img = ax.imshow(spec_db, aspect="auto", origin="lower", cmap="magma",
                    extent=[t[0], t[-1], 0, SR // 2])
    ax.set_ylim(0, 4000)
    ax.set_title("④ 原始频谱包络 SP", fontsize=11, fontweight="bold")
    ax.set_ylabel("频率 (Hz)", fontsize=9)

    # 共振峰频移后
    ax = axes[1, 1]
    spec_db_w = librosa.amplitude_to_db(sp_warped.T, ref=np.max)
    ax.imshow(spec_db_w, aspect="auto", origin="lower", cmap="magma",
              extent=[t[0], t[-1], 0, SR // 2])
    ax.set_ylim(0, 4000)
    ax.set_title(f"⑤ 共振峰频移 (factor={formant_factor:.2f})", fontsize=11, fontweight="bold")

    # 混合后频谱包络
    ax = axes[1, 2]
    spec_db_m = librosa.amplitude_to_db(sp_smooth.T, ref=np.max)
    ax.imshow(spec_db_m, aspect="auto", origin="lower", cmap="magma",
              extent=[t[0], t[-1], 0, SR // 2])
    ax.set_ylim(0, 4000)
    ax.set_title("⑥ 频谱混合+平滑 → 最终包络", fontsize=11, fontweight="bold")

    # 非周期性
    ax = axes[2, 0]
    ap_db = librosa.amplitude_to_db(ap.T, ref=np.max)
    ax.imshow(ap_db, aspect="auto", origin="lower", cmap="viridis",
              extent=[t[0], t[-1], 0, SR // 2])
    ax.set_ylim(0, 4000)
    ax.set_title("⑦ 非周期性 AP (谐波/噪声比)", fontsize=11, fontweight="bold")
    ax.set_xlabel("时间 (s)", fontsize=9)
    ax.set_ylabel("频率 (Hz)", fontsize=9)

    # 调整后非周期性
    ax = axes[2, 1]
    ap_adj_db = librosa.amplitude_to_db(ap_adj.T, ref=np.max)
    ax.imshow(ap_adj_db, aspect="auto", origin="lower", cmap="viridis",
              extent=[t[0], t[-1], 0, SR // 2])
    ax.set_ylim(0, 4000)
    ax.set_title("⑧ 非周期性调整 (男声方向)", fontsize=11, fontweight="bold")
    ax.set_xlabel("时间 (s)", fontsize=9)

    # 合成结果
    ax = axes[2, 2]
    y_conv = pw.synthesize(f0_conv, sp_smooth, ap_adj, SR)
    D_conv = librosa.amplitude_to_db(np.abs(librosa.stft(y_conv.astype(np.float64), n_fft=2048)), ref=np.max)
    ax.imshow(D_conv, aspect="auto", origin="lower", cmap="magma",
              extent=[0, len(y_conv) / SR, 0, SR // 2])
    ax.set_ylim(0, 4000)
    ax.set_title("⑨ WORLD合成 → 男声输出", fontsize=11, fontweight="bold")
    ax.set_xlabel("时间 (s)", fontsize=9)

    fig.suptitle("图9: WORLD 声码器变声管线 — 源-滤波器模型完整闭环", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig9_world_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 图9: WORLD变声管线")

# ═══════════════════════════════════════════════════════════════════════════
# 图10: 系统架构框图
# ═══════════════════════════════════════════════════════════════════════════
def fig10_architecture():
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # 定义框
    boxes = [
        (0.5, 4.5, 2.5, 1.5, "麦克风\n22kHz采样", "#dbeafe", "#1e40af"),
        (3.5, 4.5, 2.5, 1.5, "环形缓冲区\n(15秒, 线程安全)", "#d1fae5", "#065f46"),
        (6.5, 4.5, 2.5, 1.5, "滑动窗口\n(3秒 × 22kHz)", "#fef3c7", "#92400e"),
        (9.5, 4.5, 2.5, 1.5, "STFT\n(n_fft=2048)", "#ede9fe", "#5b21b6"),
        (12.5, 4.5, 2.5, 1.5, "20维频域特征\n(频谱质心/F0/...)", "#fce7f3", "#9d174d"),

        (9.5, 1.5, 2.5, 1.5, "XGBoost 分类器\n(二分类, max_depth=3)", "#fce7f3", "#9d174d"),
        (12.5, 1.5, 2.5, 1.5, "性别预测\n男 / 女 / 无语音", "#dbeafe", "#1e40af"),

        (3.5, 1.5, 2.5, 1.5, "WORLD 分析\nF0 + SP + AP", "#d1fae5", "#065f46"),
        (6.5, 1.5, 2.5, 1.5, "参数修改\nF0缩放/共振峰/频谱倾斜", "#fef3c7", "#92400e"),
        (0.5, 1.5, 2.5, 1.5, "WORLD 合成\npw.synthesize()", "#ede9fe", "#5b21b6"),
    ]

    for x, y, w, h, text, face, edge in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=face, edgecolor=edge, linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9,
                fontweight="bold", color=edge)

    # 箭头 (上排: 识别)
    arrows = [
        (3.0, 5.25, 3.5, 5.25, "#64748b"),   # 麦克风 → 缓冲区
        (6.0, 5.25, 6.5, 5.25, "#64748b"),   # 缓冲区 → 窗口
        (9.0, 5.25, 9.5, 5.25, "#64748b"),   # 窗口 → STFT
        (12.0, 5.25, 12.5, 5.25, "#64748b"), # STFT → 特征
        # 识别路径转弯
        (13.75, 4.5, 13.75, 3.0, "#64748b"),  # 特征 ↓ → 分类器
        (13.75, 3.0, 12.5, 3.0, "#64748b"),
        (12.0, 3.0, 12.5, 3.0, "#64748b"),   # 分类器 → 结果 (reversed)
        # 变声路径
        (6.0, 4.5, 6.0, 3.0, "#64748b"),   # 缓冲区 ↓ → WORLD
        (6.0, 3.0, 6.5, 3.0, "#64748b"),
        (6.0, 2.25, 6.5, 2.25, "#64748b"), # WORLD → 参数
        (3.5, 2.25, 4.0, 2.25, "#64748b"), # 参数 → 合成 (reversed, wait)
        (3.0, 2.25, 3.5, 2.25, "#64748b"),
    ]

    for x1, y1, x2, y2, color in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0,
                                    connectionstyle="arc3,rad=0"))

    # 标签
    ax.text(7.5, 6.5, "识别管线 (每1.5秒触发)", ha="center", fontsize=13,
            fontweight="bold", color="#1e40af",
            bbox=dict(boxstyle="round", facecolor="#dbeafe", alpha=0.7))
    ax.text(4.75, 0.5, "变声管线 (用户触发)", ha="center", fontsize=13,
            fontweight="bold", color="#065f46",
            bbox=dict(boxstyle="round", facecolor="#d1fae5", alpha=0.7))

    ax.set_title("图10: 系统总体架构 — 识别+变声双管线", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig10_architecture.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 图10: 系统架构")

# ═══════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("开始生成报告图表...")
    print("=" * 50)
    t_start = time.time()

    fig1_spectrograms()
    fig2_f0_contour()
    fig3_feature_importance()
    fig4_spectrum()
    fig5_confusion_matrix()
    fig6_vad_demo()
    fig7_ringbuffer()
    fig8_feature_boxplot()
    fig9_world_pipeline()
    fig10_architecture()

    elapsed = time.time() - t_start
    print(f"\n全部图表生成完成! 耗时 {elapsed:.0f}s")
    print(f"输出目录: {OUT_DIR}")
