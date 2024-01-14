# :D
from typing import Annotated
from fastapi import Depends, FastAPI
from app.routers.images import images
from app.routers.works import works
from .config.database import Base, engine, SessionLocal

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(images)
app.include_router(works)


@app.get("/")
async def root():
    return {"message": "Hello ★"}
