"""
nba_bot — NBA win probability + Polymarket edge scanner package.
"""

import logging

__version__ = "0.1.0"

# Library packages must NOT configure root logging.
# Add a NullHandler so callers who don't configure logging don't get
# "No handler found" warnings. Applications (scan.py, train.py) configure
# their own logging before calling other modules.
logging.getLogger(__name__).addHandler(logging.NullHandler())
