from fastapi import UploadFile
from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import date
from uuid import UUID

# Base
class AuthorBase(BaseModel):
    id: UUID
    name: str
    pseudonim: str
    birth_date: date
    death_date: date
    gender: Literal['MALE', 'FEMALE']
    parents: Optional[List[str]] = None
    children: Optional[List[str]] = None
    siblings: Optional[List[str]] = None

# Create
class AuthorCreate(AuthorBase):
    pass

# Read/Return
class Author(AuthorBase):
    class Config:
        from_attributes = True
