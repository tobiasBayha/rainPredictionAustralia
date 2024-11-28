from sensai.data_transformation import DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureCollector
from sensai.sklearn.sklearn_classification import SkLearnLogisticRegressionVectorClassificationModel, \
    SkLearnKNeighborsVectorClassificationModel, SkLearnRandomForestVectorClassificationModel, \
    SkLearnDecisionTreeVectorClassificationModel, SkLearnMLPVectorClassificationModel
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from .data import *
from .features import FeatureName, registry

class ModelFactory:
    COLS_USED_BY_MODELS = [COL_MINTEMP, COL_MAXTEMP, COL_RAINFALL,
                           COL_WINDGUSTSPEED, COL_WINDSPEED9AM, COL_WINDSPEED3PM,
                           COL_HUMIDITY9AM, COL_HUMIDITY3PM, COL_PRESSURE9AM, COL_PRESSURE3PM, COL_TEMP9AM,
                           COL_TEMP3PM]
    DEFAULT_FEATURES = (FeatureName.WEATHER_DEGREES, FeatureName.WEATHER_CATEGORIES, FeatureName.MINTEMP,
                        FeatureName.MAXTEMP, FeatureName.RAINFALL, FeatureName.WINDGUSTSPEED,
                        FeatureName.WINDSPEED9AM, FeatureName.WINDSPEED3PM, FeatureName.PRESSURE9AM,
                        FeatureName.PRESSURE3PM, FeatureName.TEMP9AM, FeatureName.TEMP3PM)

    @classmethod
    def create_logistic_regression_orig(cls):
        return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("LogisticRegression-orig")

    @classmethod
    def create_knn_orig(cls):
        return SkLearnKNeighborsVectorClassificationModel(n_neighbors=1) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("KNeighbors-orig")

    @classmethod
    def create_random_forest_orig(cls):
        return SkLearnRandomForestVectorClassificationModel(n_estimators=100) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("RandomForest-orig")

    @classmethod
    def create_decision_tree_orig(cls):
        return SkLearnDecisionTreeVectorClassificationModel(max_depth=2) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("DecisionTree-orig")

    @classmethod
    def create_Neural_Network_orig(cls):
        return SkLearnMLPVectorClassificationModel(solver='adam', alpha=1e-5, hidden_layer_sizes=(8, 4), random_state=1) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("NeuralNetwork-orig")

    @classmethod
    def create_logistic_regression(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
                fc.create_feature_transformer_normalisation()) \
            .with_name("LogisticRegression")

    @classmethod
    def create_random_forest(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnRandomForestVectorClassificationModel(n_estimators=100) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name("RandomForest")

    @classmethod
    def create_knn(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnKNeighborsVectorClassificationModel(n_neighbors=1) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
                                       fc.create_feature_transformer_normalisation(),
                                       DFTSkLearnTransformer(MaxAbsScaler())) \
            .with_name("KNeighbors")

    @classmethod
    def create_decision_tree(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnDecisionTreeVectorClassificationModel(max_depth=2) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
                                       fc.create_feature_transformer_normalisation(),
                                       DFTSkLearnTransformer(MaxAbsScaler())) \
            .with_name("DecisionTree")

    @classmethod
    def create_Neural_Network(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnMLPVectorClassificationModel(solver='adam', alpha=1e-5, hidden_layer_sizes=(8, 4), random_state=1) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
                                       fc.create_feature_transformer_normalisation(),
                                       DFTSkLearnTransformer(MaxAbsScaler())) \
            .with_name("NeuralNetwork")
