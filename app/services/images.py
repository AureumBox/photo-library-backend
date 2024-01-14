import uuid
import shutil
from pathlib import Path
from fastapi import Form, UploadFile, File
from sqlalchemy.orm import Session

from app.models.images import ImageModel as Model
from app.schemas import images as schemas


def get_image(db: Session, id: str):
    return db.query(Model).filter(Model.id == id).first()


def get_images(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Model).offset(skip).limit(limit).all()


def create_image(db: Session, image: schemas.ImageCreate):
    db_image = Model(id=image.id, category=image.category, img=image.img)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image


def save_image(db: Session, file: UploadFile = File(...), category: str = Form(...)):
    print("hola")
    id = uuid.uuid4()
    image_path = f"app/images/{category}/{id}.jpg"
    with Path(image_path).open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = schemas.ImageCreate.model_construct(
        id=id, category=category, img=image_path
    )
    create_image(db, image)

    return {"id": str(id), "category": category, "img": image_path}
