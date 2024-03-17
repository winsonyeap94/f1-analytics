import os
import sys
import numpy as np
import pandas as pd
import fastf1
import fastf1.plotting
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common import Config, _logger

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
fastf1.plotting.setup_mpl()
fastf1.set_log_level('ERROR')
fastf1.Cache.enable_cache(Config.FASTF1_CACHE_DIR)


class RacePaceCoefficientsLoader:
    """
    Main class for loading and processing the necessary data for calculating Race Pace Coefficients (FP and Race).
    """
    
    LAPTIME_PERCENTILE_THRESHOLD = 0.75

    def __init__(self, year, track, sessions=None, load_telemetry=True):
        self.event = fastf1.get_event(year, track)
        self.year = self.event.year
        self.track = self.event.EventName
        self.sessions = sessions or ['FP1', 'FP2', 'FP3', 'R']
        self.load_telemetry = load_telemetry
        self.intercept = dict()
        self.coefficients = dict()
        self.r2_score = dict()
        self.r2_fig = dict()
    
    def load(self):
        """
        Loads session data and fits Ridge regression models for 'FP' and 'R' session categories.
        """
        _stime = datetime.now()
        self.valid_laps_df = self._load_session_data()
        self.intercept['FP'], self.coefficients['FP'], self.r2_score['FP'], self.r2_fig['FP'] = self._fit_coefficients(self.valid_laps_df, session_category='FP')
        self.intercept['R'], self.coefficients['R'], self.r2_score['R'], self.r2_fig['R'] = self._fit_coefficients(self.valid_laps_df, session_category='R')
        _etime = datetime.now()
        _logger.info(f"[{self.year} {self.track}] Load completed in: {_etime - _stime}")
        
    def _load_session_data(self):
        """
        Loads and preprocesses the session data for a given session type.

        This function performs the following steps:
        1. Loads raw data including lap, weather, and telemetry data for the given session type.
        2. Preprocesses the data by extracting lap information, adding weather data, and filtering out invalid laps.
        3. Performs feature engineering by one-hot encoding categorical variables, splitting 'TyreLife' based on 'Compound',
        calculating 'LapNumberWithinStint', normalizing certain features, and adding 'DRS' information.
        """
        
        # ============================== Loading Raw Data ==============================
        # Lap, Weather, Telemetry Data
        laps_df = []
        weather_df = []
        telemetry_df = []
        for session in self.sessions:
            _logger.debug(f"[{self.year} {self.track}] Loading {session} lap, weather, telemetry data")
            f1_session = self.event.get_session(session)
            f1_session.load()
            append_df = f1_session.laps.assign(Session=session)
            laps_df.append(append_df)
            if self.load_telemetry:
                for driver in append_df['Driver'].sort_values().unique():
                    telemetry_df.append(f1_session.laps.pick_driver(driver).get_telemetry().assign(Driver=driver, Session=session))
            append_df = f1_session.laps.get_weather_data().drop_duplicates().sort_values(by='Time').reset_index(drop=True)\
                .assign(Session=session)
            weather_df.append(append_df)
        laps_df = pd.concat(laps_df, axis=0).reset_index(drop=True)
        weather_df = pd.concat(weather_df, axis=0).reset_index(drop=True)
        weather_df['WeatherTimeId'] = weather_df.index
        telemetry_df = pd.concat(telemetry_df, axis=0).reset_index(drop=True) if len(telemetry_df) > 0 else None
        
        # ============================== Preprocessing Data ==============================
        # Extracting lap information and adding weather
        valid_laps_df = laps_df.query("(Deleted == False) and (IsAccurate == True)").copy()
        valid_laps_df['LapTime_seconds'] = valid_laps_df['LapTime'].dt.total_seconds()
        _valid_laps_df = []
        for session in valid_laps_df['Session'].sort_values().unique():
            session_weather_df = weather_df.query(f"Session == '{session}'")
            append_df = valid_laps_df.query(f"Session == '{session}'").copy()
            append_df['WeatherTimeId'] = pd.cut(
                append_df['Time'],
                bins=session_weather_df['Time'].tolist() + [session_weather_df['Time'].max() + timedelta(days=1)],
                labels=session_weather_df['WeatherTimeId'].tolist(),
                include_lowest=True,
            )
            _valid_laps_df.append(append_df)
        valid_laps_df = pd.concat(_valid_laps_df, axis=0).reset_index(drop=True)
        valid_laps_df = valid_laps_df.merge(weather_df.drop(columns=['Time']), on=['Session', 'WeatherTimeId'], how='left')\
            .drop(columns=['WeatherTimeId'])
        valid_laps_df = valid_laps_df.query("(Rainfall == False)").copy()
        laptime_threshold = np.nanmax([
            valid_laps_df.query("Session == 'A'")['LapTime_seconds'].quantile(self.LAPTIME_PERCENTILE_THRESHOLD),
            valid_laps_df.query("Session == 'R'")['LapTime_seconds'].quantile(self.LAPTIME_PERCENTILE_THRESHOLD),
            valid_laps_df.query("Session != 'R'")['LapTime_seconds'].quantile(self.LAPTIME_PERCENTILE_THRESHOLD),
        ])
        valid_laps_df = valid_laps_df.query(f"LapTime_seconds < {laptime_threshold}").copy()
        _logger.debug(f"[{self.year} {self.track}] LapTime Threshold: {laptime_threshold:.3f} seconds")
        
        # ============================== Feature Engineering ==============================
        # One-hot encoding
        valid_laps_df = valid_laps_df.drop(columns=[x for x in valid_laps_df.columns if x.startswith('Team_')])
        valid_laps_df = valid_laps_df.drop(columns=[x for x in valid_laps_df.columns if x.startswith('Driver_')])
        valid_laps_df = valid_laps_df.drop(columns=[x for x in valid_laps_df.columns if x.startswith('Compound_')])
        valid_laps_df = pd.concat([valid_laps_df, pd.get_dummies(valid_laps_df['Team'], prefix='Team')], axis=1)
        valid_laps_df = pd.concat([valid_laps_df, pd.get_dummies(valid_laps_df['Driver'], prefix='Driver')], axis=1)
        valid_laps_df = pd.concat([valid_laps_df, pd.get_dummies(valid_laps_df['Compound'], prefix='Compound')], axis=1)

        # Split TyreLife based on Compounds
        valid_laps_df['TyreLife_SOFT'] = np.where(valid_laps_df['Compound'] == 'SOFT', valid_laps_df['TyreLife'], 0)
        valid_laps_df['TyreLife_MEDIUM'] = np.where(valid_laps_df['Compound'] == 'MEDIUM', valid_laps_df['TyreLife'], 0)
        valid_laps_df['TyreLife_HARD'] = np.where(valid_laps_df['Compound'] == 'HARD', valid_laps_df['TyreLife'], 0)

        # Lap Number within Stint
        valid_laps_df = valid_laps_df.sort_values(by=['Session', 'Driver', 'Stint', 'LapNumber'])
        valid_laps_df['LapNumberWithinStint'] = valid_laps_df.groupby(['Session', 'Driver', 'Stint']).cumcount() + 1

        # Normalisation
        for feature in ['TyreLife_SOFT', 'TyreLife_MEDIUM', 'TyreLife_HARD', 'LapNumber', 'LapNumberWithinStint']:
            if valid_laps_df[feature].nunique() == 1:
                valid_laps_df[f'{feature}_Normalised'] = 0
            else:
                valid_laps_df[f'{feature}_Normalised'] = valid_laps_df[feature] / valid_laps_df[feature].max()
        
        # Adding DRS Information to valid_laps_df
        if telemetry_df is not None:
            telemetry_df = telemetry_df.drop(columns=[x for x in telemetry_df.columns if x in ['LapNumber']])
            _telemetry_df = []
            for driver in telemetry_df['Driver'].sort_values().unique():
                driver_telemetry_df = telemetry_df.query(f"Driver == '{driver}'").copy()
                for session in telemetry_df['Session'].sort_values().unique():
                    driver_session_telemetry_df = driver_telemetry_df.query(f"Session == '{session}'").copy()
                    driver_session_laps_df = laps_df.query(f"Driver == '{driver}' and Session == '{session}'")
                    driver_session_telemetry_df['LapNumber'] = pd.cut(
                        driver_session_telemetry_df['Time'],
                        bins=driver_session_laps_df['Time'].tolist() + [driver_session_laps_df['Time'].max() + timedelta(days=1)],
                        labels=driver_session_laps_df['LapNumber'].tolist(),
                        include_lowest=True,
                    )
                    _telemetry_df.append(driver_session_telemetry_df)
            telemetry_df = pd.concat(_telemetry_df, axis=0).reset_index(drop=True)
            telemetry_df['DRS_Enabled'] = np.where(telemetry_df['DRS'].isin([10, 12, 14]), 1, 0)

            lap_drs_df = telemetry_df.groupby(['Driver', 'Session', 'LapNumber']).agg({'DRS_Enabled': 'max'}).reset_index()
            valid_laps_df = valid_laps_df.drop(columns=[x for x in valid_laps_df.columns if x in ['DRS_Enabled']])\
                .merge(lap_drs_df, on=['Driver', 'Session', 'LapNumber'], how='left')
            valid_laps_df['DRS_Enabled'] = valid_laps_df['DRS_Enabled'].fillna(0)
        
        return valid_laps_df

    def _fit_coefficients(self, valid_laps_df, session_category):
        """
        Fits a Ridge regression model to the session data and returns the model's coefficients, intercept, and a plot of predicted vs actual lap times.

        The function first checks if the session category is valid (either 'FP' or 'R'). It then prepares the input features for the model, 
        which include normalised tyre life, lap number, lap number within stint, DRS enabled status, and one-hot encoded team, driver, and compound information.
        If the session category is 'FP', the 'LapNumber_Normalised' feature is excluded. 
        
        A Ridge regression model is then fitted to the data, and the model's coefficients and intercept are extracted.
        """
        # Initialisation
        model_df = valid_laps_df.loc[valid_laps_df['Session'].str.startswith(session_category.upper()), :].copy()
        assert session_category.upper() in ['FP', 'R'], "Invalid session category. Must be either 'FP' or 'R'"
        assert not model_df.empty, "No valid data for the given session category"
        
        # Fitting linear regression and get coefficients
        input_features = ['TyreLife_SOFT', 'TyreLife_MEDIUM', 'TyreLife_HARD', 'LapNumber', 'LapNumberWithinStint', 'DRS_Enabled'] + \
            [x for x in model_df.columns if x.startswith('Team_')] + \
            [x for x in model_df.columns if x.startswith('Driver_')] + \
            [x for x in model_df.columns if x.startswith('Compound_')]
        if session_category.upper() == 'FP':
            input_features = [x for x in input_features if x not in ['LapNumber']]
        input_features = [x for x in input_features if x in model_df.columns]

        # Fit a linear regression model
        lr_model = Ridge()
        lr_model = lr_model.fit(model_df[input_features], model_df['LapTime_seconds'])

        # Extracting coefficients and intercept
        coefficients, intercept = lr_model.coef_, lr_model.intercept_
        coefficients = {k: v for k, v in zip(lr_model.feature_names_in_, coefficients)}
        coefficients_df = pd.DataFrame.from_dict(coefficients, orient='index').rename(columns={0: 'coef'})\
            .assign(abs_coef=lambda x: x['coef'].abs())\
            .sort_values('abs_coef', ascending=False)

        # Model R2 Score
        y_true, y_pred = model_df['LapTime_seconds'], lr_model.predict(model_df[input_features])
        r2_value = r2_score(y_true=model_df['LapTime_seconds'], y_pred=y_pred)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.regplot(x=y_pred, y=y_true, ax=ax)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', lw=2, color='white')
        ax.set_xlabel('Predicted Lap Time (s)')
        ax.set_ylabel('Actual Lap Time (s)')
        ax.set_title(f"Predicted vs Actual Lap Time (R2 Score: {r2_value:.3f})")

        return intercept, coefficients_df, r2_value, fig


if __name__ == "__main__":
    
    rpc_loader = RacePaceCoefficientsLoader(2021, 'Bahrain', load_telemetry=True)
    rpc_loader.load()

