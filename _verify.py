"""Quick compile + test verification."""
import compileall
import sys
import os

os.chdir(r"S:\AI\work\VRAXION_DEV")

print("=== COMPILE CHECK ===")
r1 = compileall.compile_dir("Golden Code", quiet=1)
r2 = compileall.compile_dir("Golden Draft", quiet=1)
print(f"Golden Code: {'OK' if r1 else 'FAIL'}")
print(f"Golden Draft: {'OK' if r2 else 'FAIL'}")

if not (r1 and r2):
    print("COMPILE FAILED")
    sys.exit(1)

print("\n=== ALL COMPILE CLEAN ===")
