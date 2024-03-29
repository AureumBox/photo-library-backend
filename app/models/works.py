from sqlalchemy import Column, UUID, Text, Date, ARRAY
from sqlalchemy.orm import relationship

from app.config.database import Base


class Work(Base):
    __tablename__ = "works"

    id = Column(UUID, primary_key=True)
    genre = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    meter = Column(Text, nullable=False)
    subject = Column(Text, nullable=False)
    original_language = Column(Text, nullable=False)
    elaboration_start_date = Column(Date, nullable=False)
    elaboration_end_date = Column(Date, unique=True, index=True)
    elaboration_places = Column(ARRAY(Text))

    image = relationship("Image", back_populates="work")
    publications = relationship("Publication", back_populates="work")
