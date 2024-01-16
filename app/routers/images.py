from fastapi import APIRouter, Depends, File, UploadFile, Form
from uuid import UUID
from typing import Optional
from fastapi.responses import JSONResponse

from enum import Enum
import tensorflow as tf
import numpy as np
import io
from app.image_classifier.model_specifications import image_size, class_names

from app.config.database import SessionLocal
from app.dependencies import get_db
from app.services import images as service
from app.schemas import images as schemas


images = APIRouter(prefix="/images", tags=["images"])


@images.get(
    "/",
    description="Get a list of all images",
)
def get_images(db: SessionLocal = Depends(get_db)):
    try:
        images = service.get_images(db)
        return {"images": images}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@images.get(
    "/tag/{tag}",
    description="Get a list of all images from a tag",
)
def get_images(tag: str, db: SessionLocal = Depends(get_db)):
    try:
        images = service.get_images_by_tag(db, tag)
        return {"images": images}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@images.post(
    "/classify",
    description="Classify an image with the model",
)
async def classify_image(
    file: UploadFile = File(...), db: SessionLocal = Depends(get_db)
):
    model = tf.keras.models.load_model("app/image_classifier/models/classifier1.h5")

    try:
        file_contents = await file.read()

        # Convertir los bytes a una imagen PIL
        img = tf.keras.preprocessing.image.load_img(
            io.BytesIO(file_contents), target_size=image_size
        )

        # Convertir la imagen PIL a un numpy array
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Añadir una dimensión extra al principio del array
        img_array = np.expand_dims(img_array, axis=0)

        # Normalizar los pixeles a valores entre 0 y 1
        img_array /= 255.0

        # Pasar la imagen por el modelo
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        print(predictions)
        print(predicted_class)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Image successfully classified",
                "category": f"{class_names[predicted_class]}",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


class TagEnum(str, Enum):
    animals = "animals"
    architecture = "architecture"
    battles = "battles"
    bookcovers = "bookcovers"
    book_pages = "book_pages"
    foods = "foods"
    landscapes = "landscapes"
    maps = "maps"
    paintings = "paintings"
    people = "people"
    plants = "plants"
    rivers = "rivers"
    sculptures = "sculptures"
    stamps = "stamps"


@images.post("/", description="Upload an image")
async def create_image(
    id: UUID = Form(...),
    work_id: Optional[UUID] = Form(None),
    author_id: Optional[UUID] = Form(None),
    publication_id: Optional[UUID] = Form(None),
    type: str = Form(...),
    description: str = Form(...),
    tag: TagEnum = Form(...),
    copyright: str = Form(...),
    reference: str = Form(...),
    file: UploadFile = File(...),
    db: SessionLocal = Depends(get_db),
):
    try:
        db_image = service.save_image(
            db,
            id,
            work_id,
            author_id,
            publication_id,
            type,
            description,
            tag,
            copyright,
            reference,
            file,
        )
        return {"message": "Image successfully created", "image": db_image}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
