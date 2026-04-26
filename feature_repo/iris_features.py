"""
Определение признаков для Feast по датасету Iris.
"""

from datetime import timedelta
from feast import Entity, FeatureView, FileSource, ValueType, Field
from feast.types import PrimitiveFeastType

# Определение сущности
iris_entity = Entity(
    name="iris_id",
    value_type=ValueType.INT64,
    description="Unique identifier for iris flower sample",
)

# Определение источника данных
iris_source = FileSource(
    path="../data/raw/iris_feast.parquet",
    event_timestamp_column="event_timestamp",
    timestamp_field="event_timestamp",
)

# Создание FeatureView
iris_features = FeatureView(
    name="iris_features",
    entities=[iris_entity],
)

# Привязка параметров после создания
iris_features.ttl = timedelta(days=365)
iris_features.online = True
iris_features.batch_source = iris_source

# Назначение признаков после создания
iris_features.features = [
    Field(name="sepal_length", dtype=PrimitiveFeastType.FLOAT32),
    Field(name="sepal_width", dtype=PrimitiveFeastType.FLOAT32),
    Field(name="petal_length", dtype=PrimitiveFeastType.FLOAT32),
    Field(name="petal_width", dtype=PrimitiveFeastType.FLOAT32),
    Field(name="species_encoded", dtype=PrimitiveFeastType.INT64),
]

iris_features.tags = {}