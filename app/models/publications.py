from sqlalchemy import Column, Date, Text, UUID, ForeignKey
from sqlalchemy.orm import relationship

from app.config.database import Base


class Publication(Base):
    __tablename__ = "publications"

    id = Column(UUID(as_uuid=True), primary_key=True)
    type = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    publication_date = Column(Date, nullable=False)
    publication_place = Column(Text, nullable=False)
    edition = Column(Text, nullable=False)
    publisher = Column(Text, nullable=False)
    language = Column(Text, nullable=False)
    translator = Column(Text, nullable=False)
    work_id = Column(UUID(as_uuid=True), ForeignKey("works.id"))

    image = relationship("Image", back_populates="publication")
    work = relationship("Work", back_populates="publications")
