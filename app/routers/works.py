from fastapi import APIRouter, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app.config.database import SessionLocal
from app.schemas import works as schemas
from app.dependencies import get_db
from app.services import works as service


works = APIRouter(prefix="/works", tags=["works"])


@works.get(
    "/",
    description="Get a list of all works",
)
def get_works(db: SessionLocal = Depends(get_db)):
    works = service.get_works(db)
    return {"message": "lmao World", "works": works}


@works.delete(
    "/{id}",
    description="Delete a work",
)
def delete_work(id: str, db: SessionLocal = Depends(get_db)):
    service.delete_work(db, id)
    return {"message": "Work deleted", "id": id}


@works.post(
    "/",
    description="Create a work",
)
def create_work(work: schemas.WorkCreate, db: SessionLocal = Depends(get_db)):
    work = service.create_work(db, work)
    return {"message": "lmao World", "work": work}
