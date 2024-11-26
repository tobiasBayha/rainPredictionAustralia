from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from .data import COL_RAINTOMORROW, COL_EVAPORATION, COL_DATE, COL_RAINTODAY, COL_SUNSHINE, COL_MAXTEMP, COL_LOCATION, COL_MINTEMP, COL_RAINFALL, COL_WINDGUSTSPEED, COL_WINDGUSTDIR,COL_TEMP3PM, COL_TEMP9AM, COL_WINDSPEED3PM, COL_WINDSPEED9AM, COL_HUMIDITY3PM,COL_HUMIDITY9AM, COL_PRESSURE3PM, COL_PRESSURE9AM, COL_WINDIR3PM, COL_WINDIR9AM, COL_CLOUD3PM, COL_CLOUD9AM

class ModelFactory:
    COLS_USED_BY_MODELS = [COL_MINTEMP, COL_MAXTEMP, COL_RAINFALL,
                           COL_WINDGUSTSPEED, COL_WINDSPEED9AM, COL_WINDSPEED3PM,
                           COL_HUMIDITY9AM, COL_HUMIDITY3PM, COL_PRESSURE9AM, COL_PRESSURE3PM, COL_TEMP9AM,
                           COL_TEMP3PM, COL_TEMP9AM, COL_TEMP3PM]

    @classmethod
    def create_logistic_regression_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls. COLS_USED_BY_MODELS)])),
            ("model", linear_model.LogisticRegression(solver='lbfgs', max_iter=1000))])

    @classmethod
    def create_knn_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls. COLS_USED_BY_MODELS)])),
            ("model", KNeighborsClassifier(n_neighbors=1))])

    @classmethod
    def create_random_forest_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls. COLS_USED_BY_MODELS)])),
            ("model", RandomForestClassifier(n_estimators=100))])

    @classmethod
    def create_decision_tree_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls. COLS_USED_BY_MODELS)])),
            ("model", DecisionTreeClassifier(random_state=42, max_depth=2))])

    @classmethod
    def create_Neural_Network_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls. COLS_USED_BY_MODELS)])),
            ("model", MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8, 4), random_state=1))
        ])