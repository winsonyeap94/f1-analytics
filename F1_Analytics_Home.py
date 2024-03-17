import io
import os
import fastf1
import pillow_avif
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path
from geopy.geocoders import Nominatim

st.set_page_config(
    page_title="Home", 
    page_icon=":racing_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ====================================================================================================
# Side Bar
# ====================================================================================================
with st.sidebar:
    year = st.selectbox('Pick a year to see Formula 1 events', options=list(range(2024, 2009, -1)), index=0)
    events_df = fastf1.get_event_schedule(year)
    events_df['EventDate'] = pd.to_datetime(events_df['EventDate'])
    events_df['EventFormat'] = events_df['EventFormat'].replace({
        'testing': 'Pre-Season Testing',
        'conventional': 'Conventional',
        'sprint_shootout': 'Sprint Weekend'
    })
    track = st.selectbox("Pick a track to get more detailed information", options=events_df['EventName'], index=1)

# ====================================================================================================
# Main Page 
# ====================================================================================================
st.title(':racing_car: Formula 1 Analysis')
st.divider()

# ============================== List of Events in the Year ==============================
st.header(f'Formula 1 Calendar for {year} :calendar:')


events_df = events_df[['EventDate', 'Country', 'Location', 'EventName', 'EventFormat', 'OfficialEventName']].copy()

st.dataframe(
    events_df, 
    column_config={
        'EventDate': st.column_config.DateColumn(
            "Event Date",
            format='Do MMM YYYY',
        )
    },  
)

# ============================== Track-Specific Information ==============================
st.divider()
st.header(f"Track Information for {track}")

# Getting Lat/Long
loc = Nominatim(user_agent="Geopy Library")
try:
    getLoc = loc.geocode({
        'city': events_df.query(f'EventName == "{track}"')['Location'].values[0],
        'country': events_df.query(f'EventName == "{track}"')['Country'].values[0]
    })
    latlong_df = pd.DataFrame({'Latitude': [getLoc.latitude], 'Longitude': [getLoc.longitude]})
except:
    getLoc = loc.geocode(f"""{events_df.query(f'EventName == "{track}"')['Country'].values[0]}""")
    latlong_df = pd.DataFrame({'Latitude': [getLoc.latitude], 'Longitude': [getLoc.longitude]})

# Getting images
track_to_folder_mapping = {
    "Pre-Season Testing": "Bahrain",
    "Bahrain Grand Prix": "Bahrain",
    "Saudi Arabian Grand Prix": "Saudi Arabia",
    "Australian Grand Prix": "Australia",
    "Japanese Grand Prix": "Japan",
    "Chinese Grand Prix": "China",
    "Miami Grand Prix": "US-Miami",
    "Emilia Romagna Grand Prix": "Italy-Imola",
    "Monaco Grand Prix": "Monaco",
    "Canadian Grand Prix": "Canada",
    "Spanish Grand Prix": "Spain",
    "Austrian Grand Prix": "Austria",
    "British Grand Prix": "UK",
    "Hungarian Grand Prix": "Hungary",
    "Belgian Grand Prix": "Belgium",
    "Dutch Grand Prix": "Netherlands",
    "Italian Grand Prix": "Italy-Monza",
    "Azerbaijan Grand Prix": "Azerbaijan",
    "Singapore Grand Prix": "Singapore",
    "United States Grand Prix": "US-Austin",
    "Mexico City Grand Prix": "Mexico",
    "São Paulo Grand Prix": "Brazil",
    "Las Vegas Grand Prix": "US-Las Vegas",
    "Qatar Grand Prix": "Qatar",
    "Abu Dhabi Grand Prix": "Abu Dhabi",
}
track_folder_name = track_to_folder_mapping[track]
image_folder_dir = Path(os.path.dirname(__file__), "assets", "tracks", track_folder_name)
files = os.listdir(image_folder_dir)
layout_bytesio = io.BytesIO()
layout_image = Image.open(Path(image_folder_dir, [x for x in files if x.startswith('layout')][0]))
layout_image.save(layout_bytesio, format='PNG')
layout_bytesio = layout_bytesio.getvalue()
header_bytesio = io.BytesIO()
header_image = Image.open(Path(image_folder_dir, [x for x in files if x.startswith('header')][0]))
header_image.save(header_bytesio, format='PNG')
header_bytesio = header_bytesio.getvalue()

col1, col2, col3 = st.columns(3)
with col1:
    st.image(header_bytesio)
with col2:
    st.image(layout_bytesio)
with col3:
    st.map(data=latlong_df, latitude='Latitude', longitude='Longitude', zoom=4)

