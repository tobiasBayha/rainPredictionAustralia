from typing import Tuple, Optional

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

import config

COL_DATE = 'Date'
COL_LOCATION = 'Location'
COL_MINTEMP = 'MinTemp'
COL_MAXTEMP = 'MaxTemp'
COL_RAINFALL = 'Rainfall'
COL_EVAPORATION = 'Evaporation'
COL_SUNSHINE = 'Sunshine'
COL_WINDGUSTDIR = 'WindGustDir'
COL_WINDGUSTSPEED = 'WindGustSpeed'
COL_WINDIR9AM = 'WinDir9am'
COL_WINDIR3PM = 'WinDir3pm'
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

DROP_LIST = [COL_EVAPORATION, COL_SUNSHINE, COL_CLOUD9AM, COL_CLOUD3PM]
ENCODER_LIST = [COL_DATE, COL_LOCATION]

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

    def transform_data_frame(self) -> pd.DataFrame:
        """
        :return: Transformed Dataframe
        """
        df = self.load_data_frame()
        # Map RainToday and RainTomorrow to No=0 and Yes=1
        #df[COL_RAINTODAY, COL_RAINTOMORROW]=df[COL_RAINTODAY, COL_RAINTOMORROW].map({"No":0,"Yes":1})
        df[COL_RAINTODAY] = df[COL_RAINTODAY].map({"No": 0, "Yes": 1})
        df[COL_RAINTOMORROW] = df[COL_RAINTOMORROW].map({"No": 0, "Yes": 1})
        # Drop irrelevant columns
        df=df.drop(columns=DROP_LIST)
        # Remove NAN
        df=df.dropna()
        # Change the Values of the Days Column to contain only Month as an Integer
        df[COL_DATE] = df[COL_DATE].replace(to_replace=r'\b(19\d\d|20\d\d)-\b', value='', regex=True)
        df[COL_DATE] = df[COL_DATE].replace(to_replace=r'-[0-9]{2}', value='', regex=True)
        df[COL_DATE] = df[COL_DATE].replace(to_replace=r'^[0]', value='', regex=True)
        return df

    def hot_encode_data_frame(self):
        df = self.transform_data_frame()
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_cols = encoder.fit_transform(df[ENCODER_LIST])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(), index=df.index)
        df = df.join(encoded_df)
        return df

    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresponding series of class values
        """
        df = self.hot_encode_data_frame()
        return df.drop(columns=COL_RAINTOMORROW), df[COL_RAINTOMORROW]

if __name__ == '__main__':
    # define & load dataset
    dataset = Dataset()
    X,y = dataset.load_xy()

    #Drop all categorical Data
    num_columns = []
    cat_columns = []
    for feature in X.columns:
        if X[feature].dtype != "object":
            num_columns.append(feature)
        else:
            cat_columns.append(feature)
    X = X.drop(cat_columns, axis=1)

    scaler = MinMaxScaler()
    model_X = scaler.fit(X)
    X_scaled = model_X.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, test_size=0.2, shuffle=True)

    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8, 4), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Neural Network Model Balanced Accuracy (in %):", round(balanced_accuracy_score(y_test, y_pred)*100,1))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))