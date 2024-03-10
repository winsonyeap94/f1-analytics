"""
Project Configuration File
"""

import os
from pathlib import Path


class Config:

    # ============================== DATA ==============================
    FASTF1_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/cache')
    Path(FASTF1_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # ============================== CORNER OFFSETS ==============================
    # Distances before/after a corner to consider as a corner start/end
    CORNER_START_OFFSET_M = 100
    CORNER_END_OFFSET_M = 25
