import os
import sys
import fastf1
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.common import Config
from src.data_loading import MiniSectorsLoader


st.set_page_config(
    page_title="Mini-Sector Analysis", 
    page_icon=":stopwatch:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================================================================
# Cache Functions
# ====================================================================================================
@st.cache_data
def get_events_in_year(year, only_include_past=False, exclude_pre_season_testing=True):
    events_df = fastf1.get_event_schedule(year)
    events_df['EventDate'] = pd.to_datetime(events_df['EventDate'])
    events_df['EventFormat'] = events_df['EventFormat'].replace({
        'testing': 'Pre-Season Testing',
        'conventional': 'Conventional',
        'sprint_shootout': 'Sprint Weekend'
    })
    if exclude_pre_season_testing:
        events_df = events_df.query("EventFormat != 'Pre-Season Testing'")
    if only_include_past:
        events_df = events_df.query("EventDate <= @pd.Timestamp.now()")
    return events_df


@st.cache_data
def get_drivers_in_event(track, year):
    f1_event = fastf1.get_event(year, track)
    f1_session = f1_event.get_session('FP1')
    f1_session.load()
    driver_list = f1_session.laps['Driver'].sort_values().unique().tolist()
    return driver_list


@st.cache_resource
def get_minisectors_data(track, year, sessions, drivers):
    ms_loader = MiniSectorsLoader(year, track, sessions=sessions)
    _ = ms_loader.load(drivers=drivers)
    return ms_loader


# ====================================================================================================
# Side Bar
# ====================================================================================================
with st.sidebar:
    year = st.selectbox('Pick a year', options=list(range(2024, 2009, -1)), index=0)
    
    # Get list of events in the year
    events_df = get_events_in_year(year, only_include_past=True)
    track = st.selectbox("Pick a track", options=events_df['EventName'], index=0)
    
    # Get list of drivers in the event
    with st.spinner("Loading session information from FastF1 API..."):
        driver_list = get_drivers_in_event(track, year)
    drivers_selected = st.multiselect("Pick drivers to analyse", options=driver_list, default=driver_list)
    
    # Pick sessions to analyse
    if events_df.query(f'EventName == "{track}"')['EventFormat'].values[0] == 'Sprint Weekend':
        sessions_list = ['FP1', 'Q', 'SS', 'S', 'R']
    else:
        sessions_list = ['FP1', 'FP2', 'FP3', 'Q', 'R']
    sessions_selected = st.multiselect("Pick sessions to analyse", options=sessions_list, default=['R'], format_func=lambda x: Config.SESSION_NAME_MAPPING[x])


# ====================================================================================================
# Main Page 
# ====================================================================================================
st.title(':stopwatch: Mini-Sector Analysis')
st.divider()

# Load MiniSector Data
ms_loader = get_minisectors_data(track, year, sessions_selected, drivers_selected)

minisector_metric_mapping = {
    'Speed_mean': 'Avg Speed (km/h)', 'Speed_max': 'Max Speed (km/h)', 'Speed_min': 'Min Speed (km/h)', 
    'Gear_max': 'Highest Gear', 'Gear_min': 'Lowest Gear',
}
plot_metric = st.selectbox("Pick a metric to plot", options=list(minisector_metric_mapping.keys()), index=0, format_func=lambda x: minisector_metric_mapping[x])

# Comparison of MiniSector Performance by Drivers
st.title('Mini-Sector Performance by Drivers')
minisector_stats_binned_df = ms_loader.minisector_stats_df.copy()
minisector_stats_binned_df['Speed_mean_bin'] = pd.cut(minisector_stats_binned_df['Speed_mean'], bins=range(0, 501, 25), labels=[f'{x} - {x+25}' for x in range(0, 500, 25)])
minisector_stats_binned_df = pd.pivot_table(minisector_stats_binned_df, index='Speed_mean_bin', columns='Driver', values=plot_metric, aggfunc='mean').round(0)\
    .reset_index(drop=False).rename(columns={'Speed_mean_bin': 'Average MiniSector Speed (km/h)'})
st.dataframe(
    minisector_stats_binned_df.style.background_gradient(cmap='Greens', axis=1)\
        .format({col: '{:,.0f}' for col in minisector_stats_binned_df.columns if col != 'Average MiniSector Speed (km/h)'}), 
    hide_index=True
)

# Display MiniSector Plots
fig_rows = {}
for rows in range(0, len(drivers_selected) // 3 + 1, 1):
    fig_rows[rows] = st.columns(3)
for plot_id, driver in enumerate(drivers_selected):
    tile = fig_rows[plot_id // 3][plot_id % 3].container(height=600)
    fig = ms_loader.viz_minisectors(driver, plot_var=plot_metric, cbar_lim=[ms_loader.minisector_stats_df[plot_metric].min(), ms_loader.minisector_stats_df[plot_metric].max()])
    tile.title(driver)
    tile.pyplot(fig)



