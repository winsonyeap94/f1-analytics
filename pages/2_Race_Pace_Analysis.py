import os
import sys
import fastf1
import numpy as np
import pandas as pd
from pytz import UTC
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.common import Config
from src.data_loading import RacePaceCoefficientsLoader


st.set_page_config(
    page_title="Race Pace Analysis", 
    page_icon=":checkered_flag:",
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
    for session_date in ['Session1DateUtc', 'Session2DateUtc', 'Session3DateUtc', 'Session4DateUtc', 'Session5DateUtc']:
        events_df[session_date] = pd.to_datetime(events_df[session_date], format='%Y-%m-%d %H:%M:%S')
        events_df[f"{session_date}End"] = events_df[session_date] + pd.Timedelta(hours=3)
    events_df['EventFormat'] = events_df['EventFormat'].replace({
        'testing': 'Pre-Season Testing',
        'conventional': 'Conventional',
        'sprint_shootout': 'Sprint Weekend'
    })
    if exclude_pre_season_testing:
        events_df = events_df.query("EventFormat != 'Pre-Season Testing'")
    if only_include_past:
        events_df = events_df.query("Session1DateUtcEnd <= @pd.Timestamp.now()")
    return events_df


@st.cache_data
def get_drivers_in_event(track, year):
    f1_event = fastf1.get_event(year, track)
    f1_session = f1_event.get_session('FP1')
    f1_session.load()
    driver_list = f1_session.laps['Driver'].sort_values().unique().tolist()
    return driver_list


@st.cache_resource
def get_race_pace_data(track, year, sessions):
    rpc_loader = RacePaceCoefficientsLoader(year, track, sessions=sessions, load_telemetry=True)
    rpc_loader.load()
    return rpc_loader


# ====================================================================================================
# Side Bar
# ====================================================================================================
with st.sidebar:
    year = st.selectbox('Pick a year', options=list(range(2024, 2009, -1)), index=0)
    
    # Get list of events in the year
    events_df = get_events_in_year(year, only_include_past=True)
    track = st.selectbox("Pick a track", options=events_df['EventName'], index=0)


# ====================================================================================================
# Main Page 
# ====================================================================================================
st.title(':checkered_flag: Race Pace Analysis')
st.divider()

# Load Race Pace Data
track_event_df = events_df.query("EventName == @track").copy()
sessions = ['FP1']
if track_event_df['Session2DateUtcEnd'].values[0] <= pd.Timestamp.now(tz='UTC').tz_localize(None):
    sessions.append('FP2' if track_event_df['Session2'].values[0] == 'Practice 2' else 'SS')
if track_event_df['Session3DateUtcEnd'].values[0] <= pd.Timestamp.now(tz='UTC').tz_localize(None):
    sessions.append('FP3' if track_event_df['Session3'].values[0] == 'Practice 3' else 'S')
if track_event_df['Session5DateUtcEnd'].values[0] <= pd.Timestamp.now(tz='UTC').tz_localize(None):
    sessions.append('R')
rpc_loader = get_race_pace_data(track, year, sessions=sessions)

tab1, tab2, tab3 = st.tabs(['Comparison', 'Race', 'Free Practice'])

# Comparison
with tab1:
    col1, col2 = st.columns(2)
    if rpc_loader.coefficients.get('FP', None) is not None and rpc_loader.coefficients.get('R', None) is not None:
        with col1:
            fp_r2_score, race_r2_score = rpc_loader.r2_score.get('FP', 0).round(3), rpc_loader.r2_score.get('R', 0).round(3)
            fp_intercept, race_intercept = rpc_loader.intercept.get('FP', 0).round(2), rpc_loader.intercept.get('R', 0).round(2)
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.subheader("Free Practice")
                st.metric(label='R2 Score', value=fp_r2_score)
                st.metric(label='Intercept (Base Race Pace)', value=fp_intercept)
            with sub_col2:
                st.subheader("Race")
                st.metric(label='R2 Score', value=race_r2_score, delta=round(race_r2_score - fp_r2_score, 3))
                st.metric(label='Intercept (Base Race Pace)', value=race_intercept, delta=round(race_intercept - fp_intercept, 2))
            st.subheader('Coefficients')
            coefficients = rpc_loader.coefficients.get('FP').round(2).merge(
                rpc_loader.coefficients.get('R').round(2),
                left_index=True, right_index=True,
                suffixes=('_FP', '_R')
            ).sort_values('abs_coef_R', ascending=False)\
                .assign(delta_coef=lambda x: x['coef_R'] - x['coef_FP'], abs_delta_coef=lambda x: x['abs_coef_R'] - x['abs_coef_FP'])
            st.dataframe(
                coefficients.style\
                    .background_gradient(cmap='coolwarm', subset=['coef_FP', 'coef_R'], vmin=-coefficients[['abs_coef_FP', 'abs_coef_R']].max().max(), vmax=coefficients[['abs_coef_FP', 'abs_coef_R']].max().max())\
                    .background_gradient(cmap='RdYlGn', subset=['delta_coef'], vmin=-coefficients['delta_coef'].max(), vmax=coefficients['delta_coef'].max())\
                    .format({"coef_FP": "{:.3f}", "coef_R": "{:.3f}", "delta_coef": "{:.3f}", "abs_delta_coef": "{:.3f}"}),
                height=500,
                column_config={
                    "coef_FP": st.column_config.NumberColumn(
                        "Coefficient (Free Practice)",
                        format="%.3f",
                    ),
                    "coef_R": st.column_config.NumberColumn(
                        "Coefficient (Race)",
                        format="%.3f",
                    ),
                    "abs_coef_FP": st.column_config.ProgressColumn(
                        "Abs. Coefficient (Free Practice)",
                        format="%.3f",
                        min_value=0,
                        max_value=coefficients['abs_coef_FP'].max(),
                    ),
                    "abs_coef_R": st.column_config.ProgressColumn(
                        "Abs. Coefficient (Race)",
                        format="%.3f",
                        min_value=0,
                        max_value=coefficients['abs_coef_R'].max(),
                    ),
                    "delta_coef": st.column_config.NumberColumn(
                        "Δ (Coefficient)",
                        format="%.3f",
                    ),
                    "abs_delta_coef": st.column_config.ProgressColumn(
                        "Δ (Abs. Coefficient)",
                        format="%.3f",
                        min_value=0,
                        max_value=coefficients['abs_delta_coef'].max(),
                    )
                }
            )
        with col2:
            st.subheader("Team & Driver Coefficients")
            st.markdown("#### Race")
            st.altair_chart(rpc_loader.viz_team_driver_coefficients(rpc_loader.coefficients.get('R'), rpc_loader.valid_laps_df))
            st.markdown("#### Free Practice")
            st.altair_chart(rpc_loader.viz_team_driver_coefficients(rpc_loader.coefficients.get('FP'), rpc_loader.valid_laps_df))
            st.subheader("R2 Score")
            if rpc_loader.r2_fig.get('R', None):
                st.markdown("#### Race")
                st.altair_chart(rpc_loader.r2_fig.get('R'))
            if rpc_loader.r2_fig.get('FP', None):
                st.markdown("#### Free Practice")
                st.altair_chart(rpc_loader.r2_fig.get('FP'))

# Race
with tab2:
    col1, col2 = st.columns(2)
    if rpc_loader.coefficients.get('R', None) is not None:
        with col1:
            st.metric(label='R2 Score', value=rpc_loader.r2_score.get('R', 0).round(3))
            st.metric(label='Intercept (Base Race Pace)', value=rpc_loader.intercept.get('R', 0).round(2))
            st.subheader('Coefficients')
            coefficients = rpc_loader.coefficients.get('R').round(2)
            st.dataframe(
                coefficients.style.background_gradient(cmap='coolwarm', subset='coef', vmin=-coefficients['abs_coef'].max(), vmax=coefficients['abs_coef'].max())\
                    .format({"coef": "{:.3f}"}),
                height=500,
                column_config={
                    "coef": st.column_config.NumberColumn(
                        "Coefficient",
                        format="%.3f",
                    ),
                    "abs_coef": st.column_config.ProgressColumn(
                        "Abs. Coefficient",
                        format="%.3f",
                        min_value=0,
                        max_value=coefficients['abs_coef'].max(),
                    )
                }
            )
        with col2:
            st.subheader("Team & Driver Coefficients")
            st.altair_chart(rpc_loader.viz_team_driver_coefficients(rpc_loader.coefficients.get('R'), rpc_loader.valid_laps_df))
            st.subheader("R2 Score")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                team_df = rpc_loader.valid_laps_df.copy()
                tab2_team = st.selectbox("Pick a team to highlight", options=["None"] + rpc_loader.valid_laps_df['Team'].sort_values().unique().tolist(), index=0, key="Tab2_Team")
            with sub_col2:
                if tab2_team == "None":
                    tab2_driver = None
                else:
                    team_df = team_df.query(f"Team == '{tab2_team}'").copy()
                    tab2_driver = st.radio("Pick a driver to highlight", options=['Both'] + team_df['Driver'].sort_values().unique().tolist(), index=0, horizontal=True, key="Tab2_Driver")
            _, fig = rpc_loader.viz_model_fit_accuracy(rpc_loader.model_df.get('R'), team=tab2_team, driver=tab2_driver)
            st.altair_chart(fig)


# Free Practice
with tab3:
    col1, col2 = st.columns(2)
    if rpc_loader.coefficients.get('FP', None) is not None:
        with col1:
            st.metric(label='R2 Score', value=rpc_loader.r2_score.get('FP').round(3))
            st.metric(label='Intercept (Base Race Pace)', value=rpc_loader.intercept.get('FP').round(2))
            st.subheader('Coefficients')
            coefficients = rpc_loader.coefficients.get('FP').round(2)
            st.dataframe(
                coefficients.style.background_gradient(cmap='coolwarm', subset='coef', vmin=-coefficients['abs_coef'].max(), vmax=coefficients['abs_coef'].max())\
                    .format({"coef": "{:.3f}"}),
                height=500,
                column_config={
                    "coef": st.column_config.NumberColumn(
                        "Coefficient",
                        format="%.3f",
                    ),
                    "abs_coef": st.column_config.ProgressColumn(
                        "Abs. Coefficient",
                        format="%.3f",
                        min_value=0,
                        max_value=coefficients['abs_coef'].max(),
                    )
                }
            )
        with col2:
            st.subheader("Team & Driver Coefficients")
            st.altair_chart(rpc_loader.viz_team_driver_coefficients(rpc_loader.coefficients.get('FP'), rpc_loader.valid_laps_df))
            st.subheader("R2 Score")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                team_df = rpc_loader.valid_laps_df.copy()
                tab3_team = st.selectbox("Pick a team to highlight", options=["None"] + rpc_loader.valid_laps_df['Team'].sort_values().unique().tolist(), index=0, key="Tab3_Team")
            with sub_col2:
                if tab3_team == "None":
                    tab3_driver = None
                else:
                    team_df = team_df.query(f"Team == '{tab3_team}'").copy()
                    tab3_driver = st.radio("Pick a driver to highlight", options=['Both'] + team_df['Driver'].sort_values().unique().tolist(), index=0, horizontal=True, key="Tab3_Driver")
            _, fig = rpc_loader.viz_model_fit_accuracy(rpc_loader.model_df.get('FP'), team=tab3_team, driver=tab3_driver)
            st.altair_chart(fig)


st.dataframe(rpc_loader.valid_laps_df)
