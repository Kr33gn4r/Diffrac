# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('assets/styles.css', 'assets'), ('functions/FODE_PECE.py', 'functions'), ('.venv/lib/python3.13/site-packages/dash_daq/package-info.json', 'dash_daq'), ('.venv/lib/python3.13/site-packages/dash_daq/metadata.json', 'dash_daq'), ('.venv/lib/python3.13/site-packages/dash_daq/dash_daq.min.js', 'dash_daq')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
