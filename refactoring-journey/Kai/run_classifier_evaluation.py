from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from raintomorrow.data import Dataset
from raintomorrow.model_factory import ModelFactory


if __name__ == '__main__':
    # define & load dataset
    dataset = Dataset()
    X,y = dataset.load_xy()
    y = y.to_frame()

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=True)

    # define models to be evaluated
    models = [
        ModelFactory.create_logistic_regression_orig(),
        ModelFactory.create_logistic_regression(),
        ModelFactory.create_knn_orig(),
        ModelFactory.create_knn(),
        ModelFactory.create_random_forest_orig(),
        ModelFactory.create_random_forest(),
        ModelFactory.create_decision_tree_orig(),
        ModelFactory.create_decision_tree(),
        ModelFactory.create_Neural_Network_orig(),
        ModelFactory.create_Neural_Network()
    ]

    # evaluate models
    for model in models:
        print(f"Evaluating model:\n{model}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
