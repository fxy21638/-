"""打包 VoiceGenderGUI.exe (PySide6 桌面版)"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GUI_MAIN = ROOT / "Python_Study_Local" / "voice_gender_gui.py"
VENV = Path(sys.executable).parent.parent
XGB_DIR = VENV / "Lib" / "site-packages" / "xgboost"
MODEL_DIR = Path(r"D:\new_document\Document\voice")

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onedir",
    "--noconsole",
    "--noconfirm",
    "--name", "VoiceGenderGUI",
    "--add-data", f"{GUI_MAIN.parent / 'voice_clarify.html'};.",
    "--add-data", f"{MODEL_DIR / 'voice_xgb_model.pkl'};model",
    "--add-data", f"{MODEL_DIR / 'voice_feature_names.pkl'};model",
    "--add-data", f"{MODEL_DIR / 'voice_label_mapping.pkl'};model",
    "--add-binary", f"{XGB_DIR / 'lib' / 'xgboost.dll'};xgboost/lib",
    "--add-data", f"{XGB_DIR / 'VERSION'};xgboost",
    # ── main.py 模块（GUI 通过 from main import 使用） ──
    "--hidden-import", "main",
    "--hidden-import", "fastapi",
    "--hidden-import", "fastapi.responses",
    "--hidden-import", "fastapi.middleware.cors",
    "--hidden-import", "uvicorn",
    "--hidden-import", "starlette",
    "--hidden-import", "starlette.responses",
    "--hidden-import", "starlette.middleware.cors",
    # ── 核心科学计算 ──
    "--hidden-import", "sklearn",
    "--hidden-import", "sklearn.utils._typedefs",
    "--hidden-import", "sklearn.neighbors._partition_nodes",
    "--hidden-import", "soundfile",
    "--hidden-import", "pyworld",
    "--hidden-import", "xgboost",
    "--hidden-import", "sounddevice",
    "--hidden-import", "pandas",
    "--hidden-import", "joblib",
    "--hidden-import", "scipy.signal",
    "--hidden-import", "scipy.interpolate",
    "--hidden-import", "librosa.effects",
    # ── PySide6 / matplotlib 后端 ──
    "--hidden-import", "matplotlib.backends.backend_qtagg",
    "--hidden-import", "matplotlib.backends.backend_qt5agg",
    "--hidden-import", "matplotlib.backends.qt_compat",
    "--hidden-import", "PySide6.QtCore",
    "--hidden-import", "PySide6.QtGui",
    "--hidden-import", "PySide6.QtWidgets",
    # ── 整包收集 ──
    "--collect-all", "matplotlib",
    "--collect-all", "librosa",
    "--collect-all", "scipy",
    "--collect-submodules", "sklearn",
    "--exclude-module", "tkinter",
    "--exclude-module", "_tkinter",
    "--exclude-module", "matplotlib.backends.backend_tkagg",
    "--exclude-module", "matplotlib.backends._backend_tk",
    "--distpath", str(ROOT / "dist"),
    str(GUI_MAIN),
]
subprocess.run(cmd, check=True)
print(f"\n打包完成: {ROOT / 'dist' / 'VoiceGenderGUI' / 'VoiceGenderGUI.exe'}")
