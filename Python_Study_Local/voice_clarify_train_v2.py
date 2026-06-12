"""重新训练模型：对比 纯音频 vs 音频+CSV，测试集=Actors23-24"""
import sys, os, warnings
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from main import extract_audio_features

TRAIN_AUDIO_DIR = Path(r"D:\new_document\信号与系统\信号与系统\data")
TEST_AUDIO_DIR = Path(r"D:\new_document\测试集\测试集")
CSV_PATH = Path(r"D:\new_document\Document\voice\voice.csv")
MODEL_DIR = Path(r"D:\new_document\Document\voice")

FEATURE_NAMES = [
    "meanfreq", "sd", "median", "Q25", "Q75", "IQR",
    "skew", "kurt", "sp.ent", "mode", "centroid",
    "meanfun", "minfun", "maxfun", "meandom", "maxdom",
    "dfrange", "modindx",
]
LABEL_NUM = {"male": 1, "female": 0}
LABEL_ZH = {0: "女性", 1: "男性"}


def extract_dir(audio_dir: Path, label_map: dict) -> list:
    """male/female 子目录结构"""
    rows = []
    for gender, subdir in [("male", "male"), ("female", "female")]:
        sub = audio_dir / subdir
        if not sub.exists():
            continue
        for fp in sorted(sub.glob("*.wav")):
            try:
                y, sr = sf.read(str(fp))
                if y.ndim > 1:
                    y = y.mean(axis=1)
                feats = extract_audio_features(y, sr, FEATURE_NAMES)
                d = dict(zip(FEATURE_NAMES, feats.iloc[0]))
                d["label"] = label_map[gender]
                rows.append(d)
            except Exception as e:
                print(f"  SKIP {fp.name}: {e}")
    return rows


def extract_actors(audio_dir: Path, label_map: dict) -> list:
    """Actor_XX 子目录结构"""
    rows = []
    for actor_dir in sorted(audio_dir.iterdir()):
        if not actor_dir.is_dir():
            continue
        aname = actor_dir.name
        if "23" in aname:
            gender = "male"
        elif "24" in aname:
            gender = "female"
        else:
            continue
        for fp in sorted(actor_dir.glob("*.wav")):
            try:
                y, sr = sf.read(str(fp))
                if y.ndim > 1:
                    y = y.mean(axis=1)
                feats = extract_audio_features(y, sr, FEATURE_NAMES)
                d = dict(zip(FEATURE_NAMES, feats.iloc[0]))
                d["label"] = label_map[gender]
                rows.append(d)
            except Exception as e:
                print(f"  SKIP {fp.name}: {e}")
    return rows


def train_and_eval(X_tr, y_tr, X_te, y_te, name):
    model = xgb.XGBClassifier(
        base_score=0.5, booster="gbtree", max_depth=3,
        learning_rate=0.1, n_estimators=100, random_state=0,
        objective="binary:logistic",
    )
    model.fit(X_tr, y_tr)
    tr_pred = model.predict(X_tr)
    te_pred = model.predict(X_te)
    te_proba = model.predict_proba(X_te)
    tr_acc = accuracy_score(y_tr, tr_pred)
    te_acc = accuracy_score(y_te, te_pred)
    te_f1 = f1_score(y_te, te_pred)
    cm = confusion_matrix(y_te, te_pred)
    male_acc = accuracy_score(y_te[y_te == 1], te_pred[y_te == 1]) if (y_te == 1).any() else 0
    female_acc = accuracy_score(y_te[y_te == 0], te_pred[y_te == 0]) if (y_te == 0).any() else 0

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  训练集: {len(X_tr)}条  训练准确率: {tr_acc:.3f}")
    print(f"  测试集: {len(X_te)}条  测试准确率: {te_acc:.3f}  F1: {te_f1:.3f}")
    print(f"  男声: {male_acc:.3f}  女声: {female_acc:.3f}")
    print(f"  混淆: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    # 打印前10个错误
    err_idx = np.where(te_pred != y_te)[0]
    if len(err_idx) > 0:
        print(f"  错误 ({len(err_idx)}条):")
        for i in err_idx[:8]:
            tg = "男" if y_te[i] == 1 else "女"
            pg = "男" if te_pred[i] == 1 else "女"
            print(f"    true={tg} pred={pg} F={te_proba[i][0]:.1%} M={te_proba[i][1]:.1%}")

    return model, te_acc


def main():
    print("=" * 50)
    print("1/4 提取训练音频特征 (Actors 1-22)...")
    train_rows = extract_dir(TRAIN_AUDIO_DIR, LABEL_NUM)
    print(f"   训练音频: {len(train_rows)} 条")

    print("2/4 提取测试音频特征 (Actors 23-24)...")
    test_rows = extract_actors(TEST_AUDIO_DIR, LABEL_NUM)
    print(f"   测试音频: {len(test_rows)} 条")

    test_df = pd.DataFrame(test_rows)
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df["label"].values

    # ── 方案A: 纯音频 ──
    audio_df = pd.DataFrame(train_rows)
    X_audio = audio_df[FEATURE_NAMES].values
    y_audio = audio_df["label"].values
    model_a, acc_a = train_and_eval(X_audio, y_audio, X_test, y_test, "方案A: 仅音频 (1320条)")

    # ── 方案B: 音频 + CSV ──
    print("\n3/4 加载CSV...")
    csv_df = pd.read_csv(CSV_PATH)
    csv_df["label"] = csv_df["label"].map({"male": 1, "female": 0, 1: 1, 0: 0})
    for m in [c for c in FEATURE_NAMES if c not in csv_df.columns]:
        csv_df[m] = 0.0
    csv_df = csv_df[FEATURE_NAMES + ["label"]]
    print(f"   CSV: {len(csv_df)} 条")

    combined_df = pd.concat([audio_df, csv_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=FEATURE_NAMES, keep="first")
    X_combined = combined_df[FEATURE_NAMES].values
    y_combined = combined_df["label"].values
    model_b, acc_b = train_and_eval(X_combined, y_combined, X_test, y_test,
                                     "方案B: 音频+CSV")

    # ── 选择最优方案保存 ──
    print("\n4/4 保存最优模型...")
    import joblib
    if acc_a >= acc_b:
        print(f"  选择方案A (纯音频), 准确率={acc_a:.3f}")
        best_model = model_a
    else:
        print(f"  选择方案B (音频+CSV), 准确率={acc_b:.3f}")
        best_model = model_b

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "voice_xgb_model.pkl")
    joblib.dump(FEATURE_NAMES, MODEL_DIR / "voice_feature_names.pkl")
    joblib.dump(LABEL_ZH, MODEL_DIR / "voice_label_mapping.pkl")
    print(f"   模型已保存到 {MODEL_DIR}")


if __name__ == "__main__":
    main()
