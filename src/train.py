"""
Обучение модели с логированием в MLflow.
"""
import os
import yaml
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Загрузка параметров из YAML-файла."""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    logger.info(f"Параметры загружены из {params_path}")
    return params


def train_model(
        data_dir: str = "data/processed",
        model_dir: str = "models",
        params_path: str = "params.yaml"
) -> None:
    """
    Обучение модели на подготовленно датасете с логированием в MLflow.

    Args:
        data_dir: директория с обработанными датасетами
        model_dir: директория сохранения обученной модели
        params_path: путь для файла параметров
    """
    logger.info("Начало обучения модели...")

    # Загрузка параметров
    params = load_params(params_path)
    train_params = params['train']

    # Создание директории модели
    os.makedirs(model_dir, exist_ok=True)

    # Загрузка данных
    logger.info(f"Загрузка обработанных датасетов из {data_dir}")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # Разделение на фичи и таргет
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X_train = train_df[feature_cols]
    y_train = train_df['species_encoded']
    X_test = test_df[feature_cols]
    y_test = test_df['species_encoded']

    logger.info(f"Размер Train-выборки: {X_train.shape}")
    logger.info(f"Размер Test-выборки: {X_test.shape}")

    # Установка локального источника отслеживания MLflow
    # (база данных SQLite) и имя эксперимента.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-classification")

    # Запуск нового MLflow
    with mlflow.start_run(run_name=f"{train_params['model_type']}_run"):
        logger.info("Запущен новый MLflow")

        # Создание модели в зависимости от типа
        if train_params['model_type'] == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=train_params['n_estimators'],
                max_depth=train_params['max_depth'],
                random_state=train_params['random_state']
            )
            model_name = "RandomForest"
        else:
            model = LogisticRegression(
                C=train_params['logistic_regression']['C'],
                max_iter=train_params['logistic_regression']['max_iter'],
                random_state=train_params['random_state']
            )
            model_name = "LogisticRegression"

        logger.info(f"Обучение {model_name} модели...")

        # Логирование параметров
        mlflow.log_param("model_type", train_params['model_type'])
        mlflow.log_param("random_state", train_params['random_state'])

        if train_params['model_type'] == 'random_forest':
            mlflow.log_param("n_estimators", train_params['n_estimators'])
            mlflow.log_param("max_depth", train_params['max_depth'])
        else:
            mlflow.log_param("C", train_params['logistic_regression']['C'])
            mlflow.log_param("max_iter", train_params['logistic_regression']['max_iter'])

        # Обучение модели
        model.fit(X_train, y_train)
        logger.info("Модель обучена")

        # Выполнение предсказаний
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Расчёт метрик
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Логирование метрик
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("Размер обучающей выборки", len(X_train))
        mlflow.log_metric("Размер тестовой выборки", len(X_test))

        logger.info(f"Train accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")

        # Сохранение модели
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Модель сохранена в {model_path}")

        # Сообщение для MLflow, что файл модели является артефактом эксперимента
        mlflow.log_artifact(model_path)

        # Создание и логирование report о результатах классификации  на тесте
        report = classification_report(y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(model_dir, "classification_report.csv")
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)

        # Логирование матрицы ошибок
        cm = confusion_matrix(y_test, y_pred_test)
        cm_df = pd.DataFrame(cm)
        cm_path = os.path.join(model_dir, "confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=False)
        mlflow.log_artifact(cm_path)

        # Логирование важности признаков для Random Forest
        if train_params['model_type'] == 'random_forest':
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            importance_path = os.path.join(model_dir, "feature_importance.csv")
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)

            logger.info(f"Важность признаков:\n{feature_importance}")

        # Сохранение метрик в metrics.json
        metrics = {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "model_type": model_name
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Метрики сохранены в metrics.json")

        logger.info("Логирования в MLflow завершено")
        
        # Краткая сводка
        print("\n" + "=" * 50)
        print("Краткая сводка")
        print("=" * 50)
        print(f"Модель: {model_name}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Модель сохранена в: {model_path}")
        print(f"MLflow tracking: sqlite:///mlflow.db")
        print("=" * 50 + "\n")

        print("\nКлассификация на тестовой выборке:")
        print(classification_report(y_test, y_pred_test))


if __name__ == "__main__":
    train_model()