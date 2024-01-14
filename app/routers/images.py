from fastapi import APIRouter, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse

import tensorflow as tf
import numpy as np
import io
from app.image_classifier.model_specifications import image_size, class_names

from app.config.database import SessionLocal
from app.schemas import images as schemas
from app.dependencies import get_db
from app.services import images as service


images = APIRouter(prefix="/images", tags=["images"])


@images.get(
    "/",
    description="Get a list of all images",
)
def get_images(db: SessionLocal = Depends(get_db)):
    images = service.get_images(db)
    return {"message": "lmao World", "images": images}


@images.post(
    "/classify",
    description="Classify an image with the model",
)
async def classify_image(file: UploadFile = File(...), db: SessionLocal = Depends(get_db)):
    model = tf.keras.models.load_model("app/image_classifier/models/classifier1.h5")

    try:
        file_contents = await file.read()

        # Convertir los bytes a una imagen PIL
        img = tf.keras.preprocessing.image.load_img(io.BytesIO(file_contents), target_size=image_size)

        # Convertir la imagen PIL a un numpy array
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Añadir una dimensión extra al principio del array
        img_array = np.expand_dims(img_array, axis=0)

        # Normalizar los pixeles a valores entre 0 y 1
        img_array /= 255.

        # Pasar la imagen por el modelo
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        print(predictions)
        print(predicted_class)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Imagen clasificada con éxito",
                "category": f"{class_names[predicted_class]}",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@images.post("/", description="Upload an image")
async def create_image(
    file: UploadFile = File(...),
    category: str = Form(...),
    db: SessionLocal = Depends(get_db),
):
    try:
        image = service.save_image(db, file, category)

        return JSONResponse(
            status_code=200,
            content={"message": "Imagen guardada con éxito", "image": image},
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
