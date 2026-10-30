import os
import subprocess
import sys

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen([sys.executable, os.path.join(here, "api.py")], cwd=here)
    try:
        from gui import main

        main()
    finally:
        proc.terminate()
