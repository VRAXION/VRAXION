import py_compile, sys
files = [
    r"S:\AI\work\VRAXION_DEV\Golden Code\vraxion\platinum\hallway.py",
    r"S:\AI\work\VRAXION_DEV\Golden Draft\tools\probe11_fib_volume_weight.py",
    r"S:\AI\work\VRAXION_DEV\Golden Draft\tools\probe11_dashboard.py",
]
ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK: {f}")
    except py_compile.PyCompileError as e:
        print(f"FAIL: {e}")
        ok = False
sys.exit(0 if ok else 1)
