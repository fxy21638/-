"""一键打包脚本：运行此文件即可重新生成 VoiceGender.exe"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "Python_Study_Local" / "main.py"
HTML = ROOT / "Python_Study_Local" / "voice_clarify.html"
VENV = Path(sys.executable).parent.parent  # .venv 根目录
XGB_DIR = VENV / "Lib" / "site-packages" / "xgboost"

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onedir",
    "--name", "VoiceGender",
    "--add-data", f"{HTML};.",
    "--add-binary", f"{XGB_DIR / 'lib' / 'xgboost.dll'};xgboost/lib",
    "--add-data", f"{XGB_DIR / 'VERSION'};xgboost",
    "--hidden-import", "sklearn",
    "--hidden-import", "soundfile",
    "--hidden-import", "pyworld",
    "--hidden-import", "xgboost",
    "--hidden-import", "uvicorn",
    "--distpath", str(ROOT / "dist"),
    str(MAIN),
]
subprocess.run(cmd, check=True)
print(f"\n打包完成: {ROOT / 'dist' / 'VoiceGender' / 'VoiceGender.exe'}")
