"""Allow ``python -m hifu_sim run config.yaml``."""

import sys

from .cli import main

sys.exit(main())
