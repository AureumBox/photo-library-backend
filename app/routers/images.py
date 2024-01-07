from fastapi import APIRouter, Depends
from app.config.database import SessionLocal
from app.schemas import images as schemas
from app.dependencies import get_db
from app.cruds import images as service


images = APIRouter(prefix="/images", tags=["images"])


@images.get(
    "/",
    description="Get a list of all images",
)
def get_images(db: SessionLocal = Depends(get_db)):
    images = service.get_images(db)
    return {"message": "lmao World", "images": images}


@images.post(
    "/",
    description="Upload an image",
)
def create_image(img: schemas.ImageCreate, db: SessionLocal = Depends(get_db)):
    image = service.create_image(db, img)
    return {"message": "lmao World", "image": image}
