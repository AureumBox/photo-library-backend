from enum import Enum
from pydantic import BaseModel
from typing import Optional
from uuid import UUID


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


# base
class ImageBase(BaseModel):
    id: UUID
    work_id: Optional[UUID] = None
    type: str
    description: str
    tag: TagEnum
    source: str
    copyright: str
    reference: str
    author_id: Optional[UUID] = None
    publication_id: Optional[UUID] = None


# create
class ImageCreate(ImageBase):
    pass


# read/return
class Image(ImageBase):
    class Config:
        from_attributes = True
