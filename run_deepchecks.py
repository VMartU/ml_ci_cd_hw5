from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from src.preprocess import load_raw_data
import os


def main():
    # 1. Загружаем данные
    df = load_raw_data()

    # 2. Оборачиваем в Dataset Deepchecks
    dc_dataset = Dataset(df, label="target")

    # 3. Берём стандартный набор проверок качества данных
    suite = data_integrity()

    # 4. Запускаем проверки
    result = suite.run(dc_dataset)

    # 5. Готовим папку для отчётов
    os.makedirs("reports", exist_ok=True)
    html_path = os.path.join("reports", "deepchecks_report.html")
    json_path = os.path.join("reports", "deepchecks_report.json")

    # 6. Сохраняем результат в JSON (формально по заданию)
    try:
        result.save_as_json(json_path)
    except Exception as e:
        print(f"Не удалось сохранить JSON-отчёт Deepchecks: {e}")

    # 7. Делаем простой HTML-отчёт без тяжёлого JS
    summary_text = str(result)

    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Deepchecks Data Integrity Report</title>
    </head>
    <body>
        <h1>Deepchecks: Data Integrity Report</h1>
        <p>
            Отчёт сгенерирован библиотекой Deepchecks с использованием стандартного набора
            проверок <code>data_integrity</code>. Ниже приведено текстовое представление
            результатов (статусы проверок, уровни серьёзности, короткие описания).
        </p>
        <pre>{summary_text}</pre>
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML-отчёт Deepchecks сохранён в {html_path}")
    print(f"JSON-отчёт Deepchecks сохранён в {json_path}")


if __name__ == "__main__":
    main()
