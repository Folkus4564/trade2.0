import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parents[4] / "streamlit_app" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:],
        check=True,
    )


if __name__ == "__main__":
    main()
