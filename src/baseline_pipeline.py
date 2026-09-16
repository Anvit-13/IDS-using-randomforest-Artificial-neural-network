import os
import sys

# Import and execute main pipeline from baseline_pipeline.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from baseline_pipeline import run_baseline_pipeline

if __name__ == '__main__':
    run_baseline_pipeline()
