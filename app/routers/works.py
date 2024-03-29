from fastapi import APIRouter, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app.config.database import SessionLocal
from app.schemas import works as schemas
from app.dependencies import get_db
from app.services import works as service


works = APIRouter(prefix="/works", tags=["works"])


@works.get(
    "",
    description="Get a list of all works",
)
def get_works(db: SessionLocal = Depends(get_db)):
    try:
        works = service.get_works(db)
        return {"works": works}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@works.get(
    "/{id}/publications",
    description="Get a list of all publications from a work",
)
def get_publications_by_work(id: str, db: SessionLocal = Depends(get_db)):
    try:
        publications = service.get_publications_by_work(db, id)
        return {"publications": publications}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@works.delete(
    "/{id}",
    description="Delete a work",
)
def delete_work(id: str, db: SessionLocal = Depends(get_db)):
    try:
        service.delete_work(db, id)
        return {"message": "Work successfully deleted", "id": id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@works.post(
    "",
    description="Create a work",
)
def create_work(work: schemas.WorkCreate, db: SessionLocal = Depends(get_db)):
    try:
        work = service.create_work(db, work)
        return {"message": "Work successfully created", "work": work}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
