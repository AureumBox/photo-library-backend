from fastapi import APIRouter, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app.config.database import SessionLocal
from app.schemas import authors as schemas
from app.dependencies import get_db
from app.services import authors as service


authors = APIRouter(prefix="/authors", tags=["authors"])


@authors.get(
    "",
    description="Get a list of all authors",
)
def get_authors(db: SessionLocal = Depends(get_db)):
    try:
        authors = service.get_authors(db)
        return {"authors": authors}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@authors.delete(
    "/{id}",
    description="Delete an author",
)
def delete_author(id: str, db: SessionLocal = Depends(get_db)):
    try:
        service.delete_author(db, id)
        return {"message": "Author successfully deleted", "id": id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@authors.post(
    "",
    description="Create an author",
)
def create_author(author: schemas.AuthorCreate, db: SessionLocal = Depends(get_db)):
    try:
        author = service.create_author(db, author)
        return {"message": "Author successfully created", "author": author}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
