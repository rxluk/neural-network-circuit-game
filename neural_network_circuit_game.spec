# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for neural-network-circuit-game.

Build manually:
    pip install pyinstaller
    pyinstaller neural_network_circuit_game.spec
"""

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Bundle config and track data alongside the executable
added_datas = [
    ('config.json', '.'),
    ('pista.json',  '.'),
]

# Collect matplotlib's bundled data (fonts, style sheets, etc.)
added_datas += collect_data_files('matplotlib')

a = Analysis(
    ['rede_neural_jogo.py'],
    pathex=[],
    binaries=[],
    datas=added_datas,
    hiddenimports=[
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'PIL._tkinter_finder',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # PyTorch is listed in requirements but not used at runtime
        'torch', 'torchvision', 'torchaudio',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='neural-network-circuit-game',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,   # keep console so training logs are visible
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',  # uncomment and add icon if available
)
