from sqlalchemy import Column, Date, Text, ARRAY, UUID, Enum

from app.config.database import Base

class Author(Base):
    __tablename__ = 'authors'

    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(Text)
    pseudonim = Column(Text)
    birth_date = Column(Date)
    death_date = Column(Date)
    gender = Column(Enum('MALE', 'FEMALE'))  # Asegúrate de que 'gender' es un tipo válido en tu base de datos
    parents = Column(ARRAY(Text))
    children = Column(ARRAY(Text))
    siblings = Column(ARRAY(Text))
