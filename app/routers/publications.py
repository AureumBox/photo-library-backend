from fastapi import APIRouter, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app.config.database import SessionLocal
from app.schemas import publications as schemas
from app.dependencies import get_db
from app.services import publications as service


publications = APIRouter(prefix="/publications", tags=["publications"])


@publications.get(
    "/",
    description="Get a list of all publications",
)
def get_publications(db: SessionLocal = Depends(get_db)):
    try:
        publications = service.get_publications(db)
        return {"publications": publications}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@publications.delete(
    "/{id}",
    description="Delete a publication",
)
def delete_publication(id: str, db: SessionLocal = Depends(get_db)):
    try:
        service.delete_publication(db, id)
        return {"message": "Publication successfully deleted", "id": id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@publications.post(
    "/",
    description="Create a publication",
)
def create_publication(
    publication: schemas.PublicationCreate, db: SessionLocal = Depends(get_db)
):
    try:
        publication = service.create_publication(db, publication)
        return {"message": "Publication successfully created", "publication": publication}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
