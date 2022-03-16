import os

# Obtain the parent directory of 'src'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Directory containing saved experiments
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'storage')
# Directory containing resources necessary for optimization
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
# Directory for storing resulting data (e.g., csv files), plotted figures, etc.
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# Dynamic Stats directory
DS_DIR = os.path.join(RESOURCES_DIR, 'stats')
# Profiles directory
PROF_DIR = os.path.join(RESOURCES_DIR, 'profiles')
# CallGraph directory
CG_DIR = os.path.join(RESOURCES_DIR, 'call_graphs')
# Metrics dir
METRICS_DIR = os.path.join(RESOURCES_DIR, 'metrics')

# Pickle + BZ2 suffix string
PICKLE_BZ2_SUFFIX = '.pbz2'
# Old cg refers to CGs extracted from previous program versions
CG_PREFIX = 'cg'
OLD_CG_PREFIX = 'oldcg'
# Dynamic stats prefix
DS_PREFIX = 'ds'