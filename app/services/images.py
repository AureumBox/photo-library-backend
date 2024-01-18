import uuid
import shutil
from pathlib import Path
from fastapi import Form, UploadFile, File
from sqlalchemy.orm import Session
from enum import Enum
from typing import Optional

from app.models.images import Image as Model
from app.models.works import Work
from app.models.publications import Publication
from app.models.authors import Author

from app.schemas import images as schemas


def get_image2(db: Session, id: str):
    image = db.query(Model).filter(Model.id == id).first()
    work = db.query(Work).filter(Work.id == id).first()
    publication = db.query(Publication).filter(Publication.id == id).first()
    author = db.query(Author).filter(Author.id == id).first()
    return image


def get_image(db: Session, id: str):
    image = db.query(Model).filter(Model.id == id).first()
    if not image:
        return None
    image_dict = image.__dict__
    # Get the work, publication, and author details based on their IDs
    work = (
        db.query(Work).filter(Work.id == image.work_id).first()
        if image.work_id
        else None
    )
    publication = (
        db.query(Publication).filter(Publication.id == image.publication_id).first()
        if image.publication_id
        else None
    )
    author = (
        db.query(Author).filter(Author.id == image.author_id).first()
        if image.author_id
        else None
    )

    image_dict["work"] = work.__dict__ if work else None
    image_dict["publication"] = publication.__dict__ if publication else None
    image_dict["author"] = author.__dict__ if author else None

    image_dict.pop("work_id", None)
    image_dict.pop("publication_id", None)
    image_dict.pop("author_id", None)

    return {"image": image_dict}


def get_images_by_tag(db: Session, tag: str):
    print(tag)
    return db.query(Model).filter(Model.tag == tag).all()


def get_images(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Model).offset(skip).limit(limit).all()


def create_image(db: Session, image: schemas.ImageCreate):
    db_image = Model(**image.model_dump())
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image


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


def save_image(
    db: Session,
    id: str,
    work_id: str,
    author_id: str,
    publication_id: str,
    type: str,
    description: str,
    tag: str,
    copyright: str,
    reference: str,
    file: UploadFile = File(...),
):
    image_path = f"app/images/{tag}/{id}.jpg"
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(image_path).open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    source = f"/images/{id}/file"
    image = schemas.ImageCreate.model_construct(
        id=id,
        work_id=work_id,
        author_id=author_id,
        publication_id=publication_id,
        type=type,
        description=description,
        tag=tag,
        copyright=copyright,
        reference=reference,
        source=image_path,
    )

    db_image = create_image(db, image)

    return db_image
