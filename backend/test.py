import os
from pipeline import process_image

UPLOAD_FOLDER = "backend/upload_photos"

def upload_photo():
    result  = process_image('backend/16.JPG', 'backend/output_photos')
    print(result)
    return {"status": "file saved and processed"}

upload_photo()