"""Run the CLI as ``python -m changegate <command>``, for environments without the script."""

import sys

from changegate.cli import main

if __name__ == "__main__":
    sys.exit(main())
