"""
Project Configuration File
"""

import os
from pathlib import Path


class Config:

    # ============================== DATA ==============================
    FASTF1_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/cache')
    Path(FASTF1_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
