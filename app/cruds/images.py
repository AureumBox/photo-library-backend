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
