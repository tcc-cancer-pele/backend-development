from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import yaml
from sqlalchemy import Table, Column, Integer, LargeBinary, MetaData, String, DateTime
from contextlib import asynccontextmanager
from os import getenv as osgetenv
from datetime import datetime
import databases

"""
DATABASE_URL = osgetenv("DATABASE_URL",)
database = databases.Database(DATABASE_URL)

#Initialize the API

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect() """

app = FastAPI

#Load the ML model
model = YOLO("best.pt")

#Loads the class names from the YAML file
with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)
    class_names = data['names']

#Creates an endpoint
@app.post("/reading/")
#Creates the function to analyze the image
async def read_image(file: UploadFile = File(...)):

#Reads the image sent by the user
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    #Attributes the model's result to a variable
    results = model(image, conf=0.05)

    #Print the results
    print(results[0].boxes)
    
    #Saves the image with the square drawn around the object
    results[0].save("output.jpg")

    #Define a JSON response to be returned to the user
    predictions = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            predictions.append({
            "class": class_names[cls_id],
            "confidence": round(box.conf.item()),
            "bbox": box.xyxy[0].tolist()
        })

    #Finally, return the JSON with the analysis
    return {"predictions": predictions}

#Function to save the images sent by the users in a database
metadata = MetaData()

received_images = Table(
    "received_images",
    metadata,
    Column("id", Integer, primary_key=True, nullable=False, autoincrement=True),
    Column("image", LargeBinary, nullable=False),
    Column("filename", String(255), nullable=False),
    Column("date_sent", DateTime, nullable=False, default=datetime.now)
)