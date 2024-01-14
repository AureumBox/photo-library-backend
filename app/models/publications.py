from sqlalchemy import Column, Date, Text, UUID
from app.config.database import Base

class Publication(Base):
    __tablename__ = 'publications'

    id = Column(UUID(as_uuid=True), primary_key=True)
    type = Column(Text)
    title = Column(Text)
    publication_date = Column(Date)
    publication_place = Column(Text)
    edition = Column(Text)
    publisher = Column(Text)
    language = Column(Text)
    translator = Column(Text)
    work_id = Column(UUID(as_uuid=True))
