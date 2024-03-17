import io
import os
import fastf1
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Race Pace Analysis", 
    page_icon=":checked_flag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

