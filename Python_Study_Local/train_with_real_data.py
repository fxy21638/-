"""用实际数据重新训练模型，包含房间噪音增强"""
import time, shutil
from pathlib import Path
import joblib, librosa, numpy as np, pandas as pd, soundfile as sf
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from main import extract_audio_features, _ensure_min_duration

SR = 22050
DATA_DIR = Path(r"D:\new_document\信号与系统\信号与系统\data")
MODEL_OUT = Path(r"D:\new_document\Document\voice")

# 噪音增强参数
NOISE_TYPES = ["clean", "white_low", "white_mid", "hum"]
NOISE_CONFIG = {
    "clean":       {"sigma": 0.0, "hum_freq": None},
    "white_low":   {"sigma": 0.003, "hum_freq": None},
    "white_mid":   {"sigma": 0.008, "hum_freq": None},
    "hum":         {"sigma": 0.002, "hum_freq": 50},  # 模拟电力嗡嗡声
}


def add_hum(y, freq=50, level=0.003, sr=SR):
    """添加电力频率嗡嗡声"""
    t = np.arange(len(y)) / sr
    hum = np.sin(2 * np.pi * freq * t) * level
    # 添加谐波
    hum += np.sin(2 * np.pi * freq * 2 * t) * level * 0.5
    hum += np.sin(2 * np.pi * freq * 3 * t) * level * 0.25
    return y + hum.astype(np.float64)


def load_audio(filepath):
    y, sr = sf.read(str(filepath), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float64)


def extract_features(y, feature_names):
    y = _ensure_min_duration(y, SR, min_duration=1.2)
    df = extract_audio_features(y, SR, feature_names)
    return {fn: float(df[fn].iloc[0]) for fn in feature_names}


def main():
    print("=" * 60)
    print("步骤 1: 扫描数据文件")
    print("=" * 60)

    all_files = []
    for label in ("male", "female"):
        folder = DATA_DIR / label
        if not folder.exists():
            print(f"  跳过: {folder} (不存在)")
            continue
        for fp in sorted(folder.glob("*.wav")):
            all_files.append((fp, 1 if label == "male" else 0))

    n_male = sum(1 for _, l in all_files if l == 1)
    n_female = sum(1 for _, l in all_files if l == 0)
    print(f"总文件: {len(all_files)} (男: {n_male}, 女: {n_female})")

    # 80/20 划分
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_files))
    split = int(len(all_files) * 0.8)
    train_idx = set(indices[:split])
    test_idx = set(indices[split:])

    train_files = [all_files[i] for i in train_idx]
    test_files = [all_files[i] for i in test_idx]
    print(f"训练: {len(train_files)} (男: {sum(1 for _,l in train_files if l==1)}, 女: {sum(1 for _,l in train_files if l==0)})")
    print(f"测试: {len(test_files)} (男: {sum(1 for _,l in test_files if l==1)}, 女: {sum(1 for _,l in test_files if l==0)})")

    # 获取特征名（使用当前模型的18特征）
    _, feature_names, _ = joblib.load(MODEL_OUT / "voice_feature_names.pkl"), None, None
    from main import _load_model
    _, feature_names, _ = _load_model()
    print(f"特征数: {len(feature_names)}")

    print("\n" + "=" * 60)
    print("步骤 2: 提取特征 + 噪音增强")
    print("=" * 60)

    X_train, y_train = [], []
    t0 = time.time()

    for i, (fp, label) in enumerate(train_files):
        try:
            y_clean = load_audio(fp)
            for noise_type, cfg in NOISE_CONFIG.items():
                if noise_type == "clean":
                    y_proc = y_clean
                elif cfg["hum_freq"]:
                    # 电力嗡嗡声
                    noise = np.random.randn(len(y_clean)) * cfg["sigma"]
                    y_proc = add_hum(y_clean + noise, freq=cfg["hum_freq"], level=0.005)
                else:
                    noise = np.random.randn(len(y_clean)) * cfg["sigma"]
                    y_proc = y_clean + noise

                feats = extract_features(y_proc, feature_names)
                X_train.append([feats[fn] for fn in feature_names])
                y_train.append(label)
        except Exception as e:
            pass

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(train_files)} ({elapsed:.0f}s)")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"训练样本: {len(X_train)} ({len(train_files)} 文件 × {len(NOISE_TYPES)} 增强)")

    print("\n" + "=" * 60)
    print("步骤 3: 训练模型")
    print("=" * 60)

    model = xgb.XGBClassifier(
        booster="gbtree",
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        gamma=0.2,
        learning_rate=0.06,
        max_depth=5,
        min_child_weight=2,
        n_estimators=200,
        objective="binary:logistic",
        random_state=42,
        reg_alpha=0.3,
        reg_lambda=1.0,
        subsample=0.8,
        scale_pos_weight=1.0,
    )

    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    print(f"CV Accuracy: {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")
    print(f"CV F1:       {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")

    print("全量训练...")
    model.fit(X_train, y_train)
    y_pred_tr = model.predict(X_train)
    print(f"训练集 Acc: {accuracy_score(y_train, y_pred_tr):.4f}")
    print(f"训练集 F1:  {f1_score(y_train, y_pred_tr):.4f}")

    print("\n" + "=" * 60)
    print("步骤 4: 测试集评估")
    print("=" * 60)

    X_test, y_test = [], []
    for fp, label in test_files:
        try:
            y_clean = load_audio(fp)
            feats = extract_features(y_clean, feature_names)
            X_test.append([feats[fn] for fn in feature_names])
            y_test.append(label)
        except Exception:
            pass

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f"测试集 Acc: {accuracy_score(y_test, y_pred):.4f}")
    print(f"测试集 F1:  {f1_score(y_test, y_pred):.4f}")
    if cm.shape == (2, 2):
        print(f"混淆矩阵: [[{cm[0,0]} {cm[0,1]}]  (女→女, 女→男)")
        print(f"           [{cm[1,0]} {cm[1,1]}]]  (男→女, 男→男)")
        female_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        male_recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        print(f"女声召回: {female_recall:.3f}")
        print(f"男声召回: {male_recall:.3f}")

    # 特征重要性
    print(f"\n特征重要性 Top 10:")
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    for rank, i in enumerate(idx[:10]):
        print(f"  {rank+1:2d}. {feature_names[i]:12s} = {importances[i]:.4f} ({importances[i]*100:.1f}%)")
    print(f"  Top 3 合计: {sum(importances[idx[:3]])*100:.1f}%")

    # 保存模型
    print(f"\n{'='*60}")
    print("步骤 5: 保存模型")
    print(f"{'='*60}")

    backup_dir = MODEL_OUT / "backup_v2"
    backup_dir.mkdir(exist_ok=True)
    for f in ["voice_xgb_model.pkl", "voice_feature_names.pkl", "voice_label_mapping.pkl"]:
        src = MODEL_OUT / f
        if src.exists():
            shutil.copy2(str(src), str(backup_dir / f))
    print(f"旧模型已备份到: {backup_dir}")

    joblib.dump(model, str(MODEL_OUT / "voice_xgb_model.pkl"))
    joblib.dump(feature_names, str(MODEL_OUT / "voice_feature_names.pkl"))
    joblib.dump({0: "女性", 1: "男性"}, str(MODEL_OUT / "voice_label_mapping.pkl"))
    print("新模型已保存. 请重启 GUI 测试.")

    # 测试录音
    print(f"\n{'='*60}")
    print("步骤 6: 测试你的录音")
    print(f"{'='*60}")
    rec_path = Path(r"D:\new_document\Python_Study_Local\recordings\recording.wav")
    if rec_path.exists():
        y_rec, _ = sf.read(str(rec_path), dtype="float32")
        if y_rec.ndim > 1:
            y_rec = np.mean(y_rec, axis=1)
        y_rec = y_rec.astype(np.float64)
        feats_rec = extract_features(y_rec, feature_names)
        X_rec = np.array([[feats_rec[fn] for fn in feature_names]])
        proba = model.predict_proba(X_rec)[0]
        pred = int(model.predict(X_rec)[0])
        print(f"录音预测: {'女性' if pred == 0 else '男性'} (女={proba[0]:.4f} 男={proba[1]:.4f})")
        print(f"  meanfun={feats_rec.get('meanfun',0):.4f}  sp.ent={feats_rec.get('sp.ent',0):.4f}")

    print("\n完成!")


if __name__ == "__main__":
    main()
