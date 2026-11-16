import os
import sys

# The src/ modules import each other flatly ("import paths"), so put src/ on
# the path the same way the scripts do when run directly.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
