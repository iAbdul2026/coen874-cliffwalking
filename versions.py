"""Write the exact installed versions of the packages this project uses.

  python versions.py        # prints them and writes requirements-pinned.txt (use after your final run)
Paste the Python and package versions into the README table before submission.
"""
import platform
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["gymnasium", "numpy", "scipy", "pandas", "matplotlib", "pytest"]
NOT_USED = ["torch", "stable-baselines3"]      # the brief lists these; this project does not use them

lines = []
print("Python", platform.python_version())
for p in PACKAGES:
    try:
        v = version(p); lines.append(f"{p}=={v}"); print(f"{p}=={v}")
    except PackageNotFoundError:
        print(f"{p}: NOT INSTALLED")
for p in NOT_USED:
    try:
        print(f"{p}=={version(p)} (installed, but not used by this project)")
    except PackageNotFoundError:
        print(f"{p}: not installed, not used by this project")
open("requirements-pinned.txt", "w").write("\n".join(lines) + "\n")
print("wrote requirements-pinned.txt")
