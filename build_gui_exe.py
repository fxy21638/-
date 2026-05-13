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
    "--hidden-import", "sklearn",
    "--hidden-import", "soundfile",
    "--hidden-import", "pyworld",
    "--hidden-import", "xgboost",
    "--hidden-import", "sounddevice",
    "--hidden-import", "fastapi",
    "--hidden-import", "uvicorn",
    "--hidden-import", "matplotlib.backends.backend_qtagg",
    "--hidden-import", "PySide6.QtCore",
    "--hidden-import", "PySide6.QtGui",
    "--hidden-import", "PySide6.QtWidgets",
    "--collect-submodules", "matplotlib",
    "--collect-submodules", "librosa",
    "--collect-submodules", "scipy",
    "--distpath", str(ROOT / "dist"),
    str(GUI_MAIN),
]
subprocess.run(cmd, check=True)
print(f"\n打包完成: {ROOT / 'dist' / 'VoiceGenderGUI' / 'VoiceGenderGUI.exe'}")
