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
    publications = service.get_publications(db)
    return {"message": "lmao World", "publications": publications}


@publications.delete(
    "/{id}",
    description="Delete a publication",
)
def delete_publication(id: str, db: SessionLocal = Depends(get_db)):
    service.delete_publication(db, id)
    return {"message": "publication deleted", "id": id}


@publications.post(
    "/",
    description="Create a publication",
)
def create_publication(publication: schemas.PublicationCreate, db: SessionLocal = Depends(get_db)):
    publication = service.create_publication(db, publication)
    return {"message": "lmao World", "publication": publication}
