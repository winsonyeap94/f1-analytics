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

    # ============================== MAPPINGS ==============================
    SESSION_NAME_MAPPING = {
        'FP1': 'Free Practice 1', 'FP2': 'Free Practice 2', 'FP3': 'Free Practice 3', 
        'Q': 'Qualifying', 
        'R': 'Race',
        'S': 'Sprint',
        'SS': 'Sprint Shootout',
    }
    