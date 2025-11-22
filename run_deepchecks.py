import os
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from src.preprocess import load_raw_data


def main():
    df = load_raw_data()

    # Создаём объект Dataset для deepchecks
    dc_dataset = Dataset(df, label="target")

    # Запускаем стандартный набор проверок целостности данных
    suite = data_integrity()
    result = suite.run(dc_dataset)

    os.makedirs("reports", exist_ok=True)
    output_path = os.path.join("reports", "deepchecks_report.html")

    # Сохраняем отчёт
    result.save_as_html(output_path)
    print(f"Отчёт Deepchecks сохранён в {output_path}")


if __name__ == "__main__":
    main()
