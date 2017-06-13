import sys
from cx_Freeze import setup, Executable
includes = ["matplotlib", "matplotlib.backends.backend_qt5agg"]

include_files = ["autorun.inf"]

base = None

if sys.platform == "win32":
    base = "Win32GUI"

setup(name="Power System Classifier",
      version="9000",
      description="Classifies power systems",
      options={"build_exe": {"include_files": include_files, "includes": includes}},
      executables=[Executable("Main_GUI.py", base=base)])