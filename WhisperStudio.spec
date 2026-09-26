# -*- mode: python ; coding: utf-8 -*-
# Ricetta di PyInstaller per WhisperStudio.exe (file unico, senza console).
# Uso:  pip install -r requirements-dev.txt  poi  pyinstaller WhisperStudio.spec
#
# Il motore di trascrizione va incluso esplicitamente: la release V1 ne era priva
# e l'exe restava fermo su "Inizializzazione ambiente e modelli...".
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("faster_whisper", "ctranslate2", "onnxruntime", "tokenizers", "av"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["trascrivi_locale.py"],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["torch", "tensorflow", "matplotlib", "pytest", "PIL"],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="WhisperStudio",
    console=False,
    upx=False,
)
