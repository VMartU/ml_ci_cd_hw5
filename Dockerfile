FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Обновляем pip и ставим зависимости
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# По умолчанию при запуске контейнера обучаем модель
CMD ["python", "train.py"]
