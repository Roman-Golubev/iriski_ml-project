"""
Скрипт подготавливает данные для Feast
и конвертирует CSV в Parquet с нужными столбцами времени.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def prepare_feast_data(
        input_path: str = "data/raw/iris.csv",
        output_path: str = "data/raw/iris_feast.parquet"
):
    """
    Подготавливает датасет Iris для Feast Feature Store.

    Добавляет необходимы столбцы времени и уникальные ID.
    """
    print("Подготовка данных для Feast Feature Store...")

    # Закгрузка данных
    df = pd.read_csv(input_path)

    # Добавляет уникальные ID
    df['iris_id'] = range(len(df))

    # Кодирование сортов
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['species_encoded'] = le.fit_transform(df['species'])

    # Добавление временных столбцов (для Feast)
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    df['event_timestamp'] = [base_time + timedelta(hours=i) for i in range(len(df))]
    df['created_timestamp'] = [base_time + timedelta(minutes=i) for i in range(len(df))]

    # Порядок столбцов
    df = df[[
        'iris_id',
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'species_encoded',
        'event_timestamp',
        'created_timestamp'
    ]]

    # Сохранение в Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Данные сохранены в {output_path}")
    print(f"Размер: {df.shape}")
    print(f"Столбцы: {df.columns.tolist()}")
    print(f"\nПример данных:\n{df.head()}")

    return df


if __name__ == "__main__":
    prepare_feast_data()