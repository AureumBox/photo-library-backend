from sqlalchemy import Column, Text, UUID, Enum, ForeignKey
from sqlalchemy.orm import relationship
from app.config.database import Base


class Image(Base):
    __tablename__ = "multimedia"

    id = Column(UUID(as_uuid=True), primary_key=True)
    work_id = Column(UUID(as_uuid=True), ForeignKey("works.id"))

    type = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    tag = Column(
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
        ), nullable=False
    )
    source = Column(Text, nullable=False)
    copyright = Column(Text, nullable=False)
    reference = Column(Text, nullable=False)
    author_id = Column(UUID(as_uuid=True), ForeignKey("authors.id"))
    publication_id = Column(UUID(as_uuid=True), ForeignKey("publications.id"))

    work = relationship("Work", back_populates="image")
    author = relationship("Author", back_populates="image")
    publication = relationship("Publication", back_populates="image")
