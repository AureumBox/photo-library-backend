import uuid
import shutil
from pathlib import Path
from fastapi import Form, UploadFile, File
from sqlalchemy.orm import Session

from app.models.works import Work as Model
from app.schemas import works as schemas


def get_work(db: Session, id: str):
    print(id)
    return db.query(Model).filter(Model.id == id).first()


def get_works(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Model).offset(skip).limit(limit).all()


def create_work(db: Session, work: schemas.WorkCreate):
    db_work = Model(**work.model_dump())
    db.add(db_work)
    db.commit()
    db.refresh(db_work)
    return db_work

def delete_work (db: Session, id: str):
    db.query(Model).filter(Model.id == id).delete()
    db.commit()
