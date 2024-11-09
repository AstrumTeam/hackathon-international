from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI()
PORT = 8000

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.get("/image")
def get_image():
    # Указываем путь к изображению
    image_path = "16.JPG"
    
    # Возвращаем изображение
    return FileResponse(image_path)

def start_ngrok():
    # Запускаем ngrok без параметра subdomain для бесплатного тарифа
    url = ngrok.connect(PORT)
    print(f"ngrok URL: {url}")

if __name__ == "__main__":
    start_ngrok()  # Запуск ngrok
    uvicorn.run(app, host="0.0.0.0", port=PORT)  # Запуск FastAPI