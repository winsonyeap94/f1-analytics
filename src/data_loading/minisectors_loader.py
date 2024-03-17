import os
import sys
import numpy as np
import pandas as pd
import fastf1
import fastf1.plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.collections import LineCollection

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import rotate
from common import Config, _logger

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
fastf1.plotting.setup_mpl()
fastf1.set_log_level('ERROR')
fastf1.Cache.enable_cache(Config.FASTF1_CACHE_DIR)


class MiniSectorsLoader:
    """
    Main class for calculating the Mini-Sector Performance for each driver.
    """

    def __init__(self, year, track, sessions=None):
        self.event = fastf1.get_event(year, track)
        self.year = self.event.year
        self.track = self.event.EventName
        self.sessions = sessions or ['R']

    def load(self, drivers=None):
        """
        Loads track information, driver telemetry, and aggregates minisector statistics.

        This function loads track information, driver telemetry for specified drivers and sessions, and aggregates minisector statistics. 
        If no drivers are specified, telemetry for all drivers in the sessions is loaded.
        """
        _stime = datetime.now()
        # Load track information
        _logger.info(f"[{self.year} {self.track}] Loading track information")
        self.track_df, self.corners_df, self.minisectors_df = self._load_track_info()
        
        # Load driver telemetry and assign minisectors
        _logger.info(f"[{self.year} {self.track}] Loading driver telemetry")
        _driver_telemetry_df = []
        for session in self.sessions:
            _logger.debug(f"[{self.year} {self.track}] Loading telemetry for session: {session}")
            f1_session = self.event.get_session(session)
            f1_session.load()
            if drivers is None:
                drivers = f1_session.laps['Driver'].sort_values().unique().tolist()
            for driver in drivers:
                if driver in f1_session.laps['Driver'].unique():
                    append_df = self._load_driver_telemetry(session, driver, self.minisectors_df)\
                        .assign(Session=session, Driver=driver)
                    _driver_telemetry_df.append(append_df)
        self.driver_telemetry_df = pd.concat(_driver_telemetry_df, axis=0).reset_index(drop=True)
        
        # Aggregate minisector statistics
        _logger.info(f"[{self.year} {self.track}] Aggregating minisector statistics")
        self.minisector_stats_df = self.driver_telemetry_df.groupby(['Session', 'Driver', 'MiniSectorId', 'isCorner', 'CornerNumber', 'CornerLetter'])\
            .apply(lambda x: self.agg_minisector(x)).reset_index().drop(columns=['level_6'])
        
        _etime = datetime.now()
        _logger.info(f"[{self.year} {self.track}] Load completed in {_etime - _stime}")
        
        return self.minisector_stats_df
        
    def _load_track_info(self):
        """
        Loads track information for a given session.

        This function retrieves the track information for a specified session, including the track layout, circuit corners, and mini-sectors. 
        It also performs some preprocessing on the data, such as rotating the track layout and adjusting the corner distances.
        """
        # Track Information
        session = self.event.get_session('FP1')
        session.load()
        circuit_info = session.get_circuit_info()
        
        # Generic track info
        self.track_angle_rad = circuit_info.rotation * np.pi / 180  # Converting to radians
        fastest_lap = session.laps.pick_fastest().get_telemetry()
        self.track_distance = fastest_lap['Distance'].max() / fastest_lap['RelativeDistance'].max()

        # Getting track layout (taken from any lap)
        lap = session.laps.pick_fastest()
        track_df = lap.get_pos_data()[['X', 'Y', 'Z']]
        track_df[['X_rotated', 'Y_rotated']] = rotate(track_df[['X', 'Y']].to_numpy(), angle=self.track_angle_rad)

        # Getting circuit corners
        corners_df = circuit_info.corners.copy()
        corners_df[['X_rotated', 'Y_rotated']] = rotate(corners_df[['X', 'Y']].to_numpy(), angle=self.track_angle_rad)
        corners_df['Distance_CornerStart'] = corners_df['Distance'] - Config.CORNER_START_OFFSET_M
        corners_df['Distance_CornerEnd'] = corners_df['Distance'] + Config.CORNER_END_OFFSET_M

        # Appending last and first corners to create a closed loop
        corners_df = pd.concat([
            corners_df.iloc[[-1], :].assign(keep=False, Distance=lambda x: x['Distance'] - self.track_distance, Distance_CornerStart=lambda x: x['Distance_CornerStart'] - self.track_distance, Distance_CornerEnd=lambda x: x['Distance_CornerEnd'] - self.track_distance),
            corners_df.assign(keep=True),
            corners_df.iloc[[0], :].assign(keep=False, Distance=lambda x: x['Distance'] + self.track_distance, Distance_CornerStart=lambda x: x['Distance_CornerStart'] + self.track_distance, Distance_CornerEnd=lambda x: x['Distance_CornerEnd'] + self.track_distance),
        ], axis=0)
        corners_df['__Distance_NextCorner__'] = corners_df['Distance'].shift(-1)
        corners_df['__Distance_NextCornerStart__'] = corners_df['Distance_CornerStart'].shift(-1)
        corners_df['__Distance_PrevCorner__'] = corners_df['Distance'].shift(1)
        corners_df['__Distance_PrevCornerEnd__'] = corners_df['Distance_CornerEnd'].shift(1)
        corners_df = corners_df.query("keep == True").reset_index(drop=True).drop(columns=['keep'])
        corners_df['__Distance_Corner_Corrected__'] = np.where(
            (corners_df['Distance_CornerStart'] < corners_df['__Distance_PrevCornerEnd__']) | (corners_df['Distance_CornerEnd'] > corners_df['__Distance_NextCornerStart__']),
            True, False
        )
        corners_df['Distance_CornerStart'] = np.where(
            corners_df['Distance_CornerStart'] < corners_df['__Distance_PrevCornerEnd__'],
            (corners_df['Distance'] + corners_df['__Distance_PrevCorner__']) / 2,
            corners_df['Distance_CornerStart']
        )
        corners_df['Distance_CornerEnd'] = np.where(
            corners_df['Distance_CornerEnd'] > corners_df['__Distance_NextCornerStart__'],
            (corners_df['Distance'] + corners_df['__Distance_NextCorner__']) / 2,
            corners_df['Distance_CornerEnd']
        )
        corners_df['LapCompletion'] = corners_df['Distance'] / self.track_distance
        corners_df['LapCompletion_CornerStart'] = corners_df['Distance_CornerStart'] / self.track_distance
        corners_df['LapCompletion_CornerEnd'] = corners_df['Distance_CornerEnd'] / self.track_distance
        corners_df = corners_df.drop(columns=['__Distance_NextCorner__', '__Distance_NextCornerStart__', '__Distance_PrevCorner__', '__Distance_PrevCornerEnd__', '__Distance_Corner_Corrected__'])

        # Converting the corner distances into a mini-sector
        minisectors_df = [pd.DataFrame({
            'MiniSectorId': 1,
            'LapCompletion_Start': -999,
            'isCorner': False,
            'CornerNumber': '',
            'CornerLetter': '',
        }, index=[0])]
        for idx, corner in corners_df.iterrows():
            minisectors_df.append(pd.DataFrame({
                'MiniSectorId': idx + 2,
                'LapCompletion_Start': corner['LapCompletion_CornerStart'],
                'isCorner': True,
                'CornerNumber': corner['Number'],
                'CornerLetter': corner['Letter'],
            }, index=[0]))
            minisectors_df.append(pd.DataFrame({
                'MiniSectorId': idx + 3,
                'LapCompletion_Start': corner['LapCompletion_CornerEnd'],
                'isCorner': False,
                'CornerNumber': '',
                'CornerLetter': '',
            }, index=[0]))
        minisectors_df = pd.concat(minisectors_df, axis=0).reset_index(drop=True)
        minisectors_df = minisectors_df.sort_values(['LapCompletion_Start', 'isCorner'], ascending=[True, False])\
            .drop_duplicates(subset=['LapCompletion_Start'], keep='first').reset_index(drop=True)
        minisectors_df['MiniSectorId'] = np.arange(1, len(minisectors_df) + 1)
        
        return track_df, corners_df, minisectors_df

    def _load_driver_telemetry(self, session, driver, minisectors_df):
        """
        Loads the telemetry data for a specific driver during a given session.
        """
        session = self.event.get_session(session)
        session.load()
        driver_laps = session.laps.pick_driver(driver)
        driver_telemetry_df = driver_laps.get_telemetry().query("Source == 'car'")
        driver_telemetry_df['LapNumber'] = pd.cut(
            driver_telemetry_df['Date'],
            bins=driver_laps['LapStartDate'].tolist() + [driver_laps['LapStartDate'].max() + timedelta(days=1)],
            labels=driver_laps['LapNumber'].to_list(),
            include_lowest=True,
        )
        driver_telemetry_df['LapCompletion'] = driver_telemetry_df['Distance'] % self.track_distance / self.track_distance
        driver_telemetry_df['MiniSectorId'] = pd.cut(
            driver_telemetry_df['LapCompletion'],
            bins=minisectors_df['LapCompletion_Start'].to_list() + [1],
            labels=minisectors_df['MiniSectorId'].to_list(),
            include_lowest=True,
        )
        driver_telemetry_df = driver_telemetry_df.merge(
            minisectors_df[['MiniSectorId', 'isCorner', 'CornerNumber', 'CornerLetter']], on='MiniSectorId', how='left'
        )
        return driver_telemetry_df

    @staticmethod
    def agg_minisector(df):
        return pd.DataFrame({
            'Speed_min': df['Speed'].min(),
            'Speed_mean': df['Speed'].mean(),
            'Speed_max': df['Speed'].max(),
            'Gear_min': df['nGear'].min(),
            'Gear_max': df['nGear'].max(),
        }, index=[0])

    def viz_minisectors(self, driver, sessions=None, plot_var='Speed_mean', cbar_lim=None, offset_vector=None):
        """
        Visualise minisector performance for a given stat.
        """
        sessions = sessions or self.sessions
        assert plot_var in ['Speed_min', 'Speed_mean', 'Speed_max', 'Gear_min', 'Gear_max'], f"Invalid `plot_var` value: '{plot_var}'"
        
        # Creating an artificial telemetry data with the average speed across each minisector
        minisector_lap_df = self.driver_telemetry_df.query("(LapNumber == 1) and (Driver == @driver)")[['Session', 'Driver', 'MiniSectorId', 'X', 'Y', 'Z', 'Distance', 'LapCompletion']]\
            .merge(self.minisector_stats_df.query("Driver == @driver"), on=['Session', 'Driver', 'MiniSectorId'], how='left')
        if minisector_lap_df is not None:
            minisector_lap_df = minisector_lap_df.query("Session.isin(@sessions)").copy()
        minisector_lap_df[['X_rotated', 'Y_rotated']] = rotate(minisector_lap_df[['X', 'Y']].to_numpy(), angle=self.track_angle_rad)
        # Aggregating across Sessions (if there are multiple)
        minisector_lap_df = minisector_lap_df.groupby(['Driver', 'MiniSectorId', 'X', 'Y', 'Z', 'X_rotated', 'Y_rotated', 'Distance', 'LapCompletion', 'isCorner', 'CornerNumber', 'CornerLetter'])\
            .mean(numeric_only=True)\
            .sort_values(['Driver', 'MiniSectorId', 'Distance'])\
            .reset_index(drop=False)
        # Adding the first minisector to the end to close the loop
        minisector_lap_df = pd.concat([
            minisector_lap_df,
            minisector_lap_df.iloc[[0], :]
        ], axis=0).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(20, 8))
        fig.suptitle(f"{self.track} ({self.year})\nMinisector Analysis ({plot_var}) - {driver}", fontsize=16)
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.05)
        ax.axis('off')
        ax.plot(minisector_lap_df['X_rotated'], minisector_lap_df['Y_rotated'], color='white', linewidth=16, zorder=0)

        # Adding the average speed across each minisector
        cmap = mpl.cm.plasma
        points = np.array([minisector_lap_df['X_rotated'], minisector_lap_df['Y_rotated']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(minisector_lap_df[plot_var].min(), minisector_lap_df[plot_var].max())
        lc = LineCollection(
            # [minisector_lap_df[['X_rotated', 'Y_rotated']].to_numpy()],
            segments,
            cmap=cmap, norm=norm, linestyle='-', linewidth=8, zorder=1
        )
        lc.set_array(minisector_lap_df[plot_var])
        line = ax.add_collection(lc)  # Merge all line segments together

        # Adding colorbar
        cb_axes = fig.add_axes([0.25, 0.03, 0.5, 0.03])
        if cbar_lim is None:
            normlegend = mpl.colors.Normalize(vmin=minisector_lap_df[plot_var].min(), vmax=minisector_lap_df[plot_var].max())
        else:
            normlegend = mpl.colors.Normalize(vmin=cbar_lim[0], vmax=cbar_lim[1])
        legend = mpl.colorbar.ColorbarBase(cb_axes, cmap=cmap, norm=normlegend, orientation='horizontal')

        # Adding corner labels
        offset_vector = offset_vector or [1000, 0]  # offset length is chosen arbitrarily to 'look good'
        for _, corner in self.corners_df.iterrows():
            txt = f"{corner['Number']}{corner['Letter']}"
            offset_angle = corner['Angle'] * np.pi / 180
            offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
            text_x, text_y = corner['X'] + offset_x, corner['Y'] + offset_y
            text_x, text_y = rotate([text_x, text_y], angle=self.track_angle_rad)
            track_x, track_y = rotate([corner['X'], corner['Y']], angle=self.track_angle_rad)
            ax.scatter(text_x, text_y, color='grey', s=300, zorder=2)
            ax.plot([track_x, text_x], [track_y, text_y], color='grey', linewidth=2, zorder=2)
            ax.text(text_x, text_y, txt, fontsize=10, color='white', ha='center', va='center_baseline', zorder=2)
        
        return fig


if __name__ == "__main__":
    
    ms_loader = MiniSectorsLoader(2024, 'Saudi Arabia', sessions=['FP3', 'R'])
    _ = ms_loader.load(drivers=['VER', 'LEC'])
    ms_loader.viz_minisectors('VER', plot_var='Speed_mean')
    ms_loader.viz_minisectors('LEC', sessions=['R'], plot_var='Speed_mean')
