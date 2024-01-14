from pydantic import BaseModel
from datetime import date
from uuid import UUID

class PublicationBase(BaseModel):
    id: UUID
    type: str
    title: str
    publication_date: date
    publication_place: str
    edition: str
    publisher: str
    language: str
    translator: str
    work_id: UUID

class PublicationCreate(PublicationBase):
    pass

class Publication(PublicationBase):
    class Config:
        from_attributes = True
