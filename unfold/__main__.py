"""Allows `python -m unfold …` to invoke the converter."""

import sys

from .convert import main

if __name__ == "__main__":
    sys.exit(main())
