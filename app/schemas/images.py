from pydantic import BaseModel

# base
class ImageBase(BaseModel):
    id: str
    category: str | None = None
    img: str | None = None


# create
class ImageCreate(ImageBase):
    pass


# read/return
class Image(ImageBase):
    pass

    class Config:
        orm_mode = True
