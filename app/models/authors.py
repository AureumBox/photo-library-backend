from sqlalchemy import Column, Date, Text, ARRAY, UUID

from app.config.database import Base

class Authors(Base):
    __tablename__ = 'authors'

    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(Text)
    pseudonim = Column(Text)
    birth_date = Column(Date)
    death_date = Column(Date)
    gender = Column(Text)  # Asegúrate de que 'gender' es un tipo válido en tu base de datos
    parents = Column(ARRAY(Text))
    children = Column(ARRAY(Text))
    siblings = Column(ARRAY(Text))
