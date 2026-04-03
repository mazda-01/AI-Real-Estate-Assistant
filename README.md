# 🏡 AI Real Estate Assistant (Vision & RAG)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-lightgrey.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-yellow.svg)

## 📌 О проекте
**ИИ-Ассистент по недвижимости** — это микросервисное приложение, объединяющее технологии **Компьютерного зрения (Computer Vision)** и **генерации текста с учетом поиска (RAG)**, для предоставления интеллектуальных инструментов в сфере недвижимости. Проект демонстрирует навыки интеграции современных моделей (LLM и детекция объектов) для создания удобного пользовательского продукта.

Сервис состоит из двух основных функций:
1. **📸 Классификация изображений (YOLOv11):** Обнаружение и классификация объектов на загружаемых фотографиях с помощью Ultralytics YOLOv11.
2. **💬 RAG-Ассистент по недвижимости:** ИИ-чат-бот, который служит экспертом рынка недвижимости. Он находит релевантные объявления из векторной базы данных (Qdrant) и обрабатывает их с помощью мощной LLM (Llama 3) через Groq, чтобы давать точные, основанные на реальных данных и интересные ответы.

## 🚀 Основные возможности
- **Детекция объектов**: Быстрая обработка изображений с использованием YOLOv11 и эндпоинтов FastAPI.
- **RAG-конвейер (Пайплайн)**:
   - Собственный парсер для сбора актуальных данных о недвижимости (с Циан).
   - Векторизация текста с использованием мультиязычной модели `HuggingFace Embeddings`.
   - Контекстный поиск с применением фреймворка `LangChain` и векторной БД `Qdrant`.
   - Интеллектуальная генерация ответов через `Groq (Llama-3.3-70b-versatile)`.
- **Интерактивный веб-интерфейс**: Чистый и отзывчивый фронтенд, построенный на Streamlit.
- **Микросервисная архитектура**: Четкое разделение между FastAPI (бэкенд) и Streamlit (фронтенд).

## 🛠️ Стек технологий
- **Backend**: Python, FastAPI, Uvicorn, Pandas
- **Machine Learning & AI**: Ultralytics (YOLOv11), LangChain, HuggingFace (`sentence-transformers`)
- **LLM Провайдер**: API Groq
- **Векторная БД**: Qdrant
- **Сбор данных (Парсинг)**: BeautifulSoup, Requests
- **Frontend**: Streamlit

## 📂 Структура проекта
```text
.
├── backend/
│   ├── main.py              # FastAPI сервер, эндпоинты и логика ML-моделей
│   ├── parser.py            # Собственный парсер данных недвижимости
│   ├── qdrant.ipynb         # Jupyter-ноутбук для настройки векторной БД
│   ├── data.csv             # Собранный датасет (информация о квартирах)
│   ├── yolo11n.pt           # Веса модели YOLO
│   └── requirements.txt     # Зависимости бэкенда
├── frontend/
│   ├── app.py               # Код интерфейса приложения Streamlit
│   └── requirements.txt     # Зависимости фронтенда
└── README.md                # Документация проекта
```

## ⚙️ Установка и запуск

1. **Клонируйте репозиторий:**
```bash
git clone git@github.com:mazda-01/AI-Real-Estate-Assistant.git
cd AI-Real-Estate-Assistant
```

2. **Настройка переменных окружения:**
Создайте файл `.env` в директории `backend/` со следующими ключами:
```env
HF_TOKEN=ваш_токен_huggingface
GROQ_API_KEY=ваш_ключ_groq
QDRANT_URL=ваш_url_qdrant
QDRANT_API_KEY=ваш_ключ_qdrant
```

3. **Запуск Бэкенда (FastAPI):**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

4. **Запуск Фронтенда (Streamlit):**
*Откройте новую вкладку терминала.*
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

*(Примечание: Если запуск происходит на удаленной облачной виртуальной машине, убедитесь, что вы обновили `VM_PUBLIC_IP` в файле `frontend/app.py` перед запуском фронтенда).*

