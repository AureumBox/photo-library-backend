from sqlalchemy import Boolean, Column, Enum, String, UUID

from app.config.database import Base


class ImageModel(Base):
    __tablename__ = "images"

    id = Column(UUID, primary_key=True, index=True)
    category = Column(
        Enum(
            "animals",
            "architecture",
            "battles",
            "bookcovers",
            "book_pages",
            "foods",
            "landscapes",
            "maps",
            "paintings",
            "people",
            "plants",
            "rivers",
            "sculptures",
            "stamps",
            name="img_category",
        ),
        nullable=False,
    )
    img = Column(String, nullable=False)