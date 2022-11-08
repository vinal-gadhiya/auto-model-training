import shutil
from typing import List

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates

from data_visualize import number_of_images


app = FastAPI()
templates = Jinja2Templates(directory="templates/")

@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse('index.html', context={"request": request})

@app.post('/upload')
async def upload_file(request: Request, files: List[UploadFile] = File(None)):
    for img in files:
        with open(f'{img.filename}', 'wb') as buffer:
            shutil.copyfileobj(img.file, buffer)
    # return {"file_name": img.filename}
    return templates.TemplateResponse('index.html', context={"request": request})

@app.get('/visualize_data')
async def visualize_data(request: Request):
    obj1 = number_of_images("fastapi-test-folder")
    no_of_img = obj1.get_number_of_images
    return templates.TemplateResponse('data_vis_temp.html', context={"request": request, "no_img": obj1.get_number_of_images()})