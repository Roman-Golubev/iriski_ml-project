# Проект iriski_ml-project

## Цель  
Целью проекта является выполнение классификации ирисов с использованием DVC, MLflow и Feast.  
Для задачи классификации в проекте применяются модели RandomForestClassifier и LogisticRegression.


## Краткая инструкция
**1 Скачать проект по ссылке [https://github.com/Roman-Golubev/iriski_ml-project].**   
**2 Запустить Docker.**  
**3 Проект создан для версии python3.11.**  
**4 Находясь в виртуальной среде, в корне проекта выполнить команду: `pip install -r requirements.txt`.**  
**5 Инициализовать dvc `dvc init` и логировать данные об используемом датасете `dvc add data/raw/iris.csv`.**  
**6 Чтобы избежать передачи датасета в git выполнить `git add data/raw/.gitignore data/raw/iris.csv.dvc`.**  
**7 Для запуска ml-модели выполнить `dvc repro`.**  
**8 Для просмотра результатов эксперимента выполнить `mlflow ui --backend-store-uri sqlite:///mlflow.db` и открыть [http://localhost:5000] в браузере.**  
**9 Для передачи признаков в Feast перейти в папку feature_repo и выполнить команду `feast apply`.**  
**10 Материализовать признаки в онлайн-хранилище - команда `feast materialize 2026-01-01T00:00:00 2026-12-31T23:59:59`.**  
**11 Открыть UI Feast - выполнить команду `feast ui` и открыть [http://localhost:8888] в браузере.**     


## Реализуемый пайплайн MLOps
(Docker, Postgres) -> venv -> DVC(train -> test -> (data save, model save)) -> (mlflow ui, feast ui)