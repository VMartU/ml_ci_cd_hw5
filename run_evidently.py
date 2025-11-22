import os
from sklearn.model_selection import train_test_split
from src.preprocess import load_raw_data
from evidently.report import Report
from evidently.metrics import DataDriftTable


def main():
    df = load_raw_data()

    # Разделим данные на "reference" и "current"
    ref, curr = train_test_split(df, test_size=0.5, random_state=42)

    # Искусственно создадим дрейф по одному признаку
    # (например, увеличим длину чашелистика на 20%)
    feature_to_drift = df.columns[0]  # первая колонка признаков
    curr[feature_to_drift] = curr[feature_to_drift] * 1.2

    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=ref, current_data=curr)

    os.makedirs("reports", exist_ok=True)
    output_path = os.path.join("reports", "evidently_drift_report.html")
    report.save_html(output_path)

    print(f"Отчёт Evidently по дрейфу данных сохранён в {output_path}")


if __name__ == "__main__":
    main()
