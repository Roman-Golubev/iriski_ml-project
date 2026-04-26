"""
Определение признаков для Feastпо датасету Iris.
"""
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float64, Int64


# Определение сущностей
iris_entity = Entity(
    name="iris_id",
    value_type=ValueType.INT64,
    description="Unique identifier for iris flower sample"
)


# Определение источника данных
iris_source = FileSource(
    path="../data/raw/iris_feast.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)


# Определение набора признаков (FeatureView)
iris_features = FeatureView(
    name="iris_features",
    entities=["iris_id"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="sepal_length", dtype=Float64),
        Feature(name="sepal_width", dtype=Float64),
        Feature(name="petal_length", dtype=Float64),
        Feature(name="petal_width", dtype=Float64),
        Feature(name="species_encoded", dtype=Int64),
    ],
    online=True,
    batch_source=iris_source,
    tags={},
)