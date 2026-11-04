"""python -m icfr_agent_evals.scanners rewrites docs/evidence/scout-scanners.diff."""

import sys

from icfr_agent_evals.scanners._upstream import main

sys.exit(main())
