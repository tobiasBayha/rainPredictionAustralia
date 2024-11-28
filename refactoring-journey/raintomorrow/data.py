from typing import Tuple, Optional

import pandas as pd

from . import config

COL_DATE = 'Date'
COL_LOCATION = 'Location'
COL_MINTEMP = 'MinTemp'
COL_MAXTEMP = 'MaxTemp'
COL_RAINFALL = 'Rainfall'
COL_EVAPORATION = 'Evaporation'
COL_SUNSHINE = 'Sunshine'
COL_WINDGUSTDIR = 'WindGustDir'
COL_WINDGUSTSPEED = 'WindGustSpeed'
COL_WINDIR9AM = 'WindDir9am'
COL_WINDIR3PM = 'WindDir3pm'
COL_WINDSPEED9AM = 'WindSpeed9am'
COL_WINDSPEED3PM = 'WindSpeed3pm'
COL_HUMIDITY9AM = 'Humidity9am'
COL_HUMIDITY3PM = 'Humidity3pm'
COL_PRESSURE9AM = 'Pressure9am'
COL_PRESSURE3PM = 'Pressure3pm'
COL_CLOUD9AM = 'Cloud9am'
COL_CLOUD3PM = 'Cloud3pm'
COL_TEMP9AM = 'Temp9am'
COL_TEMP3PM = 'Temp3pm'
COL_RAINTODAY = 'RainToday'
COL_RAINTOMORROW = 'RainTomorrow'

COLS_WEATHER_CATEGORIES = [COL_LOCATION, COL_WINDGUSTDIR, COL_WINDIR9AM, COL_WINDIR3PM, COL_RAINTODAY]
COLS_WEATHER_DEGREES = [COL_HUMIDITY3PM, COL_HUMIDITY9AM, ]
COLS_DROP_LIST = [COL_EVAPORATION, COL_SUNSHINE, COL_CLOUD9AM, COL_CLOUD3PM]


class Dataset:
    def __init__(self, num_samples: Optional[int] = None, random_seed: int =42):

        self.num_samples = num_samples
        self.random_seed = random_seed

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full DataFrame with all columns
        """
        df=pd.read_csv(config.csv_data_path())
        return df

    def data_frame_cleanup(self) -> pd.DataFrame:
        """
        :return: a cleaned up DataFrame with less missing Data
        """
        df=self.load_data_frame()
        df=df.drop(columns=COLS_DROP_LIST)
        return df.dropna()

    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresponding series of class values
        """
        #df=self.load_data_frame()
        df = self.data_frame_cleanup()
        return df.drop(columns=COL_RAINTOMORROW), df[COL_RAINTOMORROW]