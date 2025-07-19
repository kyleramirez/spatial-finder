#!/usr/bin/env python3
"""
Audible Tools Entry Point
Activates virtual environment if needed and calls main.py with all arguments.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    """Main entry point for audible_tools."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    venv_path = script_dir / "virtualenv"
    main_py = script_dir / "main.py"

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)

    # If virtual environment exists and we're not in it, activate it
    if venv_path.exists() and not in_venv:
        # Use the virtual environment's Python
        if os.name == "nt":  # Windows
            python_exe = venv_path / "Scripts" / "python.exe"
        else:  # Unix-like
            python_exe = venv_path / "bin" / "python"

        if python_exe.exists():
            # Re-run with the virtual environment's Python
            cmd = [str(python_exe), str(main_py)] + sys.argv[1:]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)
        else:
            print(f"Error: Virtual environment Python not found at {python_exe}")
            print("Please run 'python -m venv virtualenv' and 'pip install -r requirements.txt' first")
            sys.exit(1)
    else:
        # We're already in a virtual environment or no venv exists, run main.py directly
        # try:
        # Import and run main.py
        import main

        main.cli()
        # except ImportError as e:
        #     print(f"Error importing main.py: {e}")
        #     print("Please make sure you've installed the required dependencies:")
        #     print("  pip install -r requirements.txt")
        #     sys.exit(1)
        # except Exception as e:
        #     print(f"Error running main.py: {e}")
        #     sys.exit(1)


if __name__ == "__main__":
    main()
