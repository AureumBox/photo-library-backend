import uuid
import shutil
from fastapi import Form, File
from sqlalchemy.orm import Session

from app.models.publications import Publication as Model
from app.schemas import publications as schemas


def get_publication(db: Session, id: str):
    print(id)
    return db.query(Model).filter(Model.id == id).first()


def get_publications(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Model).offset(skip).limit(limit).all()


def create_publication(db: Session, publication: schemas.PublicationCreate):
    db_publication = Model(**publication.model_dump())
    db.add(db_publication)
    db.commit()
    db.refresh(db_publication)
    return db_publication

def delete_publication (db: Session, id: str):
    db.query(Model).filter(Model.id == id).delete()
    db.commit()
 
