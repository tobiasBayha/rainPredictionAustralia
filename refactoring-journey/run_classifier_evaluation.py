from sensai.evaluation import ClassificationEvaluatorParams, ClassificationModelEvaluation
from sensai.util import logging

from raintomorrow.data import Dataset
from raintomorrow.model_factory import ModelFactory


def main():
    # define & load dataset
    dataset = Dataset()
    io_data = dataset.load_io_data()

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

    # declare parameters to be used for evaluation, i.e. how to split the data (fraction and random seed)
    evaluator_params = ClassificationEvaluatorParams(fractional_split_test_fraction=0.3,
                                                     fractional_split_random_seed=42,
                                                     binary_positive_label=dataset.class_positive)

    # use a high-level utility class for evaluating the models based on these parameters
    ev = ClassificationModelEvaluation(io_data, evaluator_params=evaluator_params)
    ev.compare_models(models, fit_models=True)

if __name__ == '__main__':
    logging.run_main(main)
