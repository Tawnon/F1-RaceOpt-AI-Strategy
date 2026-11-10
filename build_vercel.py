from pathlib import Path
import shutil


ROOT_DIR = Path(__file__).resolve().parent


def main():
    source = ROOT_DIR / "static"
    destination = ROOT_DIR / "public" / "static"
    shutil.copytree(source, destination, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
