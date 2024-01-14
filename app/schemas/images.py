from fastapi import UploadFile
from pydantic import BaseModel


# base
class ImageBase(BaseModel):
    id: str
    category: str | None = None
    img: str | None = None


# create
class ImageCreate(ImageBase):
    pass


# return after classifying
class ImageClassified():
    category: str | None = None


# create in db
"""
class ImageCreate(ImageBase):
    file: UploadFile
"""


# read/return
class Image(ImageBase):
    pass

    class Config:
        from_attributes = True
