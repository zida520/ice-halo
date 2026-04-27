# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 增加递归深度限制（处理 PyTorch 等大型库）
sys.setrecursionlimit(10000)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('static', 'static'),   # 前端页面文件夹
        ('model', 'model'),     # 模型和预处理器文件夹
    ],
    hiddenimports=[
        'torch',
        'torch._C',
        'torch.nn',
        'flask',
        'flask_cors',
        'sklearn',
        'sklearn.preprocessing',
        'joblib',
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FoodTimePredictor',      # 生成的 exe 名称（不含 .exe）
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,                  # 保留控制台窗口，便于查看日志（发布时可改为 False）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)