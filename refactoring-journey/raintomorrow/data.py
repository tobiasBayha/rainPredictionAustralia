import logging
from typing import Optional

import pandas as pd
from sensai import InputOutputData
from sensai.util.string import ToStringMixin

from . import config

log = logging.getLogger(__name__)

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

CLASS_RAIN_TOMORROW = 'Yes'
CLASS_NO_RAIN_TOMORROW = 'No'

COLS_WEATHER_CATEGORIES = [COL_LOCATION, COL_WINDGUSTDIR, COL_WINDIR9AM, COL_WINDIR3PM, COL_RAINTODAY]
COLS_WEATHER_DEGREES = [COL_HUMIDITY3PM, COL_HUMIDITY9AM, ]
COLS_DROP_LIST = [COL_EVAPORATION, COL_SUNSHINE, COL_CLOUD9AM, COL_CLOUD3PM]


class Dataset(ToStringMixin):
    def __init__(self, num_samples: Optional[int] = None, random_seed: int =42):

        self.num_samples = num_samples
        self.random_seed = random_seed
        self.class_positive = CLASS_RAIN_TOMORROW
        self.class_negative = CLASS_NO_RAIN_TOMORROW

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the dataframe with removed columns and removed missing data
        """

        csv_path = config.csv_data_path()
        log.info(f"Loading {self} from {csv_path}")
        df=pd.read_csv(csv_path).dropna()

        #df = df.drop(columns=COLS_DROP_LIST)
        #df = df.dropna()

        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)
        return df


    def load_io_data(self) -> InputOutputData:
        """
        :return: the I/O data
        """
        return InputOutputData.from_data_frame(self.load_data_frame(), COL_RAINTOMORROW)