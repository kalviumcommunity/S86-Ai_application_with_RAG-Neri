from pathlib import Path
import subprocess
import sys


def get_python_executable():
    project_root = Path(__file__).resolve().parent
    venv_win = project_root / ".venv" / "Scripts" / "python.exe"
    venv_unix = project_root / ".venv" / "bin" / "python"

    if venv_win.exists():
        return str(venv_win)
    elif venv_unix.exists():
        return str(venv_unix)

    return sys.executable


def main():
    python_exe = get_python_executable()
    print(f"Starting Neri using {python_exe}...")
    print("")

    api_process = subprocess.Popen(
        [
            python_exe,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )

    try:
        subprocess.run(
            [
                python_exe,
                "-m",
                "streamlit",
                "run",
                "app/streamlit_app.py",
            ],
            check=True,
        )
    finally:
        api_process.terminate()
        api_process.wait()


if __name__ == "__main__":
    main()