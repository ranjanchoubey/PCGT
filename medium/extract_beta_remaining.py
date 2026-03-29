"""Extract β for remaining 5 datasets: deezer, co-cs, co-physics, am-comp, am-photo"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse main extract_beta but only run from deezer onward
from extract_beta import CONFIGS, main as full_main
import extract_beta

# Override CONFIGS to only include remaining datasets
REMAINING = ['deezer-europe', 'coauthor-cs', 'coauthor-physics', 'amazon-computers', 'amazon-photo']
extract_beta.CONFIGS = [c for c in CONFIGS if c[0] in REMAINING]

if __name__ == '__main__':
    full_main()
