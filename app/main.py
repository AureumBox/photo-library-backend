# :D
from typing import Annotated
from fastapi import Depends, FastAPI
from app.routers.images import images
from app.routers.works import works
from app.routers.authors import authors
from app.routers.publications import publications
from .config.database import Base, engine, SessionLocal
from fastapi.middleware.cors import CORSMiddleware
# Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(images)
app.include_router(works)
app.include_router(authors)
app.include_router(publications)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello ★"}
