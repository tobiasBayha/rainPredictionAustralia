from enum import Enum

from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns

from .data import *


class FeatureName(Enum):
    WEATHER_DEGREES = 'weather_degrees'
    WEATHER_CATEGORIES = 'weather_categories'

    #COL_DATE = 'Date'
    MINTEMP = 'MinTemp'
    MAXTEMP = 'MaxTemp'
    RAINFALL = 'Rainfall'
    WINDGUSTSPEED = 'WindGustSpeed'
    WINDSPEED9AM = 'WindSpeed9am'
    WINDSPEED3PM = 'WindSpeed3pm'
    PRESSURE9AM = 'Pressure9am'
    PRESSURE3PM = 'Pressure3pm'
    TEMP9AM = 'Temp9am'
    TEMP3PM = 'Temp3pm'


registry = FeatureGeneratorRegistry()

registry.register_factory(FeatureName.WEATHER_DEGREES, lambda: FeatureGeneratorTakeColumns(COLS_WEATHER_DEGREES,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)))

registry.register_factory(FeatureName.WEATHER_CATEGORIES, lambda: FeatureGeneratorTakeColumns(COLS_WEATHER_CATEGORIES,
    categorical_feature_names=COLS_WEATHER_CATEGORIES))

registry.register_factory(FeatureName.MINTEMP, lambda: FeatureGeneratorTakeColumns(COL_MINTEMP,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.MAXTEMP, lambda: FeatureGeneratorTakeColumns(COL_MAXTEMP,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.RAINFALL, lambda: FeatureGeneratorTakeColumns(COL_RAINFALL,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.WINDGUSTSPEED, lambda: FeatureGeneratorTakeColumns(COL_WINDGUSTSPEED,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.WINDSPEED9AM, lambda: FeatureGeneratorTakeColumns(COL_WINDSPEED9AM,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.WINDSPEED3PM, lambda: FeatureGeneratorTakeColumns(COL_WINDSPEED3PM,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.PRESSURE9AM, lambda: FeatureGeneratorTakeColumns(COL_PRESSURE9AM,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.PRESSURE3PM, lambda: FeatureGeneratorTakeColumns(COL_PRESSURE3PM,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.TEMP9AM, lambda: FeatureGeneratorTakeColumns(COL_TEMP9AM,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.TEMP3PM, lambda: FeatureGeneratorTakeColumns(COL_TEMP3PM,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))