"""
Подготовка данных.
Разделение на сеты train/test и базовый препроцессинг.
"""
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Загрузка параметров из YAML-файла."""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    logger.info(f"Параметры загружены из {params_path}")
    return params


def prepare_data(
        input_path: str = "data/raw/iris.csv",
        output_dir: str = "data/processed",
        params_path: str = "params.yaml"
) -> None:
    """
    Подготовка датасета.

    Args:
        input_path: путь к необработанному датасету
        output_dir: директория для сохранения обработанного датасета
        params_path: путь для файла параметров
    """
    logger.info("Начало подготовки данных...")

    # Загрузка параметров
    params = load_params(params_path)
    test_size = params['prepare']['test_size']
    random_state = params['prepare']['random_state']

    # Создание директории для сохранения обработанного датасета
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка данных
    logger.info(f"Загрузка датасета из {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Размер датасета: {df.shape}")

    # Базовые статитсики
    logger.info(f"Распределение сортов:\n{df['species'].value_counts()}")

    # Кодирование целевой переменной
    le = LabelEncoder()
    df['species_encoded'] = le.fit_transform(df['species'])

    # Сохранение таблицы соответствия классов
    classes_df = pd.DataFrame({
        'species': le.classes_,
        'encoded': range(len(le.classes_))
    })
    classes_df.to_csv(os.path.join(output_dir, "label_classes.csv"), index=False)
    logger.info(f"Классы таргета сохранены в {output_dir}/label_classes.csv")

    # Сплитование
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Размер обучающей выборки: {len(X_train)}")
    logger.info(f"Размер тестовой выборки: {len(X_test)}")

    # Сохранение обработанных данных
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    logger.info(f"Обработанные датасеты сохранены в {output_dir}")
    logger.info("Подготовка данных успешно завершена!")

    # Краткая сводка
    print("\n" + "=" * 50)
    print("Краткая сводка")
    print("=" * 50)
    print(f"Необработанный датасет: {input_path}")
    print(f"Директория с результатами: {output_dir}")
    print(f"Размер тестовой выборки: {test_size}")
    print(f"Random state: {random_state}")
    print(f"Train-сэмплы: {len(X_train)}")
    print(f"Test-сэмплы: {len(X_test)}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    prepare_data()