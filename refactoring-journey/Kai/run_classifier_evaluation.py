#Dataset prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Import Models
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#Model Validation
from sklearn.metrics import classification_report, confusion_matrix


#Projec reference
from raintomorrow.data import (Dataset, COL_RAINTOMORROW, COL_EVAPORATION, COL_DATE, COL_RAINTODAY, COL_SUNSHINE,
                               COL_MAXTEMP, COL_LOCATION, COL_MINTEMP, COL_RAINFALL, COL_WINDGUSTSPEED, COL_WINDGUSTDIR,
                               COL_TEMP3PM, COL_TEMP9AM, COL_WINDSPEED3PM, COL_WINDSPEED9AM, COL_HUMIDITY3PM,
                               COL_HUMIDITY9AM, COL_PRESSURE3PM, COL_PRESSURE9AM, COL_WINDIR3PM, COL_WINDIR9AM,
                               COL_CLOUD3PM, COL_CLOUD9AM)

def main():
    # define & load dataset
    dataset = Dataset()
    X,y = dataset.load_xy()

    # project columns used by the models
    cols_used_by_models = [COL_MINTEMP, COL_MAXTEMP, COL_RAINFALL,
                           COL_WINDGUSTSPEED, COL_WINDSPEED9AM, COL_WINDSPEED3PM,
                           COL_HUMIDITY9AM, COL_HUMIDITY3PM, COL_PRESSURE9AM, COL_PRESSURE3PM, COL_TEMP9AM,
                           COL_TEMP3PM,COL_TEMP9AM, COL_TEMP3PM]
    X = X[cols_used_by_models]


    scaler = StandardScaler()
    model_X = scaler.fit(X)
    X_scaled = model_X.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, test_size=0.2, shuffle=True)

    # define models to be evaluated
    models = [
        linear_model.LogisticRegression(solver='lbfgs', max_iter=1000),
        KNeighborsClassifier(n_neighbors=1),
        RandomForestClassifier(n_estimators=100),
        DecisionTreeClassifier(random_state=42, max_depth=2),
        MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8, 4), random_state=1)
    ]

    # evaluate models
    for model in models:
        print(f"Evaluating model:\n{model}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()