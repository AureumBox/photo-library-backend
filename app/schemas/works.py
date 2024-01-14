from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
from datetime import date

# base
class WorkBase(BaseModel):
    id: UUID
    genre: str
    title: str
    description: str
    meter: str
    subject: str
    original_language: str
    elaboration_start_date: date
    elaboration_end_date: date
    elaboration_places: Optional[List[str]]

# create
class WorkCreate(WorkBase):
    pass

# read/return
class Work(WorkBase):
    class Config:
        from_attributes = True
