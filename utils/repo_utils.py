import os

def ensure_repo_setup():
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('runs', exist_ok=True)    

    