from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI

app = FastAPI()
PORT = 8000

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

def start_ngrok():
    url = ngrok.connect(PORT)
    print(f"ngrok URL: {url}")

if __name__ == "__main__":
    start_ngrok()
    uvicorn.run(app, host="0.0.0.0", port=PORT)