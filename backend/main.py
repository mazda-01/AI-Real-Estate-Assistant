import logging
from contextlib import asynccontextmanager
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd

from PIL import Image
import io

from ultralytics import YOLO
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cross encoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

from dotenv import load_dotenv
import os

# Подгружаем ключи
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


logger = logging.getLogger('uvicorn.info')

yolo_model = None
llm = None
rag_chain = None

# Для ответа юзеру
class ImageClass(BaseModel):
    class_name: str
    class_index: int

# Вопрос юзера
class TextInput(BaseModel):
    question: str

# Ответ ЛЛмки юзеру
class LLMAnswer(BaseModel):
    text: str

def format_docs(docs):
    lines = []
    for d in docs:
        meta = d.metadata
        lines.append(
            f"ЦЕНА: {meta.get('price', '—')}\n"
            f"КОМНАТ: {meta.get('rooms', '—')}\n"
            f"ЭТАЖ: {meta.get('floor', '—')}\n"
            f"ПЛОЩАДЬ: {meta.get('total_area_m2', '—')}\n"
            f"ССЫЛКА: {meta.get('url', '—')}"
        )
    return "\n" + "\n\n".join(lines) + "\n"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для инициализации и завершения работы FastAPI приложения.
    Загружает модели машинного обучения при запуске приложения и удаляет их после завершения.
    """
    global yolo_model
    global rag_chain
    global llm

    def get_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    embeddings_model = get_embeddings()
    
    def load_vector_store():
        client = QdrantClient(
                url=QDRANT_URL, 
                api_key=QDRANT_API_KEY
            )
        
        return QdrantVectorStore(
                client=client,
                collection_name="cian",
                embedding=embeddings_model
            )

    vector_store = load_vector_store()

    # Загрузка Yolo модели
    yolo_model = YOLO("yolo11n.pt")
    logger.info('YOLO model loaded')

    # Загрузка LLM модели
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=2000, groq_api_key=GROQ_API_KEY)
    logger.info('LLM loaded')

    rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты эксперт-аналитик рынка недвижимости с многолетним опытом и отличным чувством юмора! 🎥
    Твоя задача — проанализировать предоставленные квартиры и выдать приблизительную цену.
    
    ПРАВИЛА ОТВЕТА:
    - Отвечай на том языке, на котором к тебе обратился пользователь (если на русском — отвечай на русском, если на английском - отвечай на английском).
    - Проводи глубокий анализ.
    - Подмечай забавные особенности (завышенные цены).
    - Структурируй ответ с эмодзи.
    
    Стиль анализа: Юмор должен быть смешным и слегка саркастичным. Цель — сделать анализ интересным и главное ТОЧНЫМ!
    
    ОБЯЗАТЕЛЬНО: Сначала напиши свои мысли в тегах <think>...</think>, затем дай итоговый ответ. Используй только предоставленный контекст базы."""),
    ("human", """📺 ДАННЫЕ ДЛЯ АНАЛИЗА:
    {context}

    🎯 ЗАПРОС НА ЭКСПЕРТИЗУ: {question}""")
    ])



    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt | llm | StrOutputParser()
        )
    logger.info("Application startup complete: All models loaded.")

    yield
    # Удаление моделей и освобождение ресурсов
    del yolo_model, rag_chain, llm



app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    """
    Возвращает приветственное сообщение при обращении к корневому маршруту API.
    """
    return 'Hello, user!'

@app.post('/clf_image')
async def classify_image(file: UploadFile):
    """
    Эндпоинт для классификации изображений.
    Принимает файл изображения, обрабатывает его, делает предсказание и возвращает название и индекс класса.
    """
    global yolo_model
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = yolo_model(image, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        raise HTTPException(status_code=404, detail="На изображении объекты не найдены")

    top_idx = boxes.conf.argmax().item()
    top_class = int(boxes.cls[top_idx].item())
    class_name = yolo_model.names[top_class]

    return ImageClass(
            class_name=class_name,
            class_index=top_class
        )

@app.post('/rag')
async def rag(data: TextInput):
    """
    Эндпоинт для RAG
    """
    global rag_chain

    question = data.question.strip()
    answer = await rag_chain.ainvoke(question)

    return LLMAnswer(text=answer.strip())

if __name__ == "__main__":
    # Запуск приложения на localhost с использованием Uvicorn
    # производится из командной строки: python your/path/api/main.py
    uvicorn.run("main:app", host='130.193.57.190', port=8000, reload=True)