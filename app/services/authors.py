import uuid
import shutil
from fastapi import Form, File
from sqlalchemy.orm import Session

from app.models.authors import Author as Model
from app.schemas import authors as schemas


def get_author(db: Session, id: str):
    print(id)
    return db.query(Model).filter(Model.id == id).first()


def get_authors(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Model).offset(skip).limit(limit).all()


def create_author(db: Session, author: schemas.AuthorCreate):
    db_author = Model(**author.model_dump())
    db.add(db_author)
    db.commit()
    db.refresh(db_author)
    return db_author

def delete_author (db: Session, id: str):
    db.query(Model).filter(Model.id == id).delete()
    db.commit()

