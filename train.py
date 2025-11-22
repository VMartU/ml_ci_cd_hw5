import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from src.preprocess import load_raw_data, split_data
from src.model import create_model
import joblib


def main():
    # 1. Загружаем данные и делим
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Настройка эксперимента MLflow
    mlflow.set_experiment("iris_random_forest_experiment")

    n_estimators = 100
    max_depth = 4
    random_state = 42

    with mlflow.start_run():
        # 3. Создаём и обучаем модель
        model = create_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        # 4. Предсказания и метрики
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 5. Логируем параметры и метрики в MLflow
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", acc)

        # 6. Сохраняем модель как артефакт
        os.makedirs("artifacts", exist_ok=True)
        model_path = os.path.join("artifacts", "rf_iris_model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model_artifacts")

        # 7. Логируем модель через MLflow (как "sk_model")
        mlflow.sklearn.log_model(model, artifact_path="sk_model")

        print(f"Accuracy: {acc:.4f}")
        print("Модель и метрики залогированы в MLflow.")


if __name__ == "__main__":
    main()
