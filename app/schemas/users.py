from pydantic import BaseModel


# base
class UserBase(BaseModel):
    email: str


# create
class UserCreate(UserBase):
    password: str


# read/return
class User(UserBase):
    id: int
    is_active: bool

    class Config:
        orm_mode = True
