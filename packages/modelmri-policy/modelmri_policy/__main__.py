"""`python -m modelmri_policy` / the `modelmri-policy` script."""

from __future__ import annotations

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
