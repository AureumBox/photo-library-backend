from fastapi import APIRouter, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
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
def classify_image(file: UploadFile = File(...), db: SessionLocal = Depends(get_db)):
    try:
        return {"message": "lmao World", "image": "hola"}
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
