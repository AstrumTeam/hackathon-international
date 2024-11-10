from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
from pathlib import Path
from pipeline import process_image

app = FastAPI()
PORT = 8000

# Папки для загрузки и обработки файлов
UPLOAD_FOLDER = "backend/upload_photos"
OUTPUT_FOLDER = "backend/output_photos"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

@app.post("/upload_photo/")
async def upload_photo(file: UploadFile = File(...)):
    # Сохраняем загруженный файл
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Вызываем функцию обработки изображения
        result = process_image(file_path, OUTPUT_FOLDER)
        recognized_text = result["recognized_text"]
        processed_image_path = result["output_image_path"]

        # Возвращаем JSON с текстом и URL обработанного изображения
        return JSONResponse(content={
            "recognized_text": recognized_text,
            "processed_image_url": f"/get_processed_image/{Path(processed_image_path).name}"
        })

    except Exception as e:
        # Логируем и отправляем сообщение об ошибке
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during image processing")

# Маршрут для получения обработанного изображения
@app.get("/get_processed_image/{image_name}")
async def get_processed_image(image_name: str):
    processed_image_path = os.path.join(OUTPUT_FOLDER, image_name)
    if os.path.exists(processed_image_path):
        return FileResponse(processed_image_path, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=404, detail="Processed image not found")

# Функция для запуска ngrok туннеля
def start_ngrok():
    url = ngrok.connect(PORT)
    print(f"ngrok URL: {url}")

if __name__ == "__main__":
    start_ngrok()  # Запуск ngrok для получения публичного URL
    uvicorn.run(app, host="0.0.0.0", port=PORT)  # Запуск FastAPI