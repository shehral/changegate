"""Run the record commands: ``python -m changegate.record verify record.json --key pub.pem``."""

import sys

from changegate.record.cli import main

if __name__ == "__main__":
    sys.exit(main())
