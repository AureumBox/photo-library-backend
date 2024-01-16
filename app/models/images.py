from sqlalchemy import Column, Text, UUID, Enum
from app.config.database import Base


class Image(Base):
    __tablename__ = "multimedia"

    id = Column(UUID(as_uuid=True), primary_key=True)
    work_id = Column(UUID(as_uuid=True))
    type = Column(Text)
    description = Column(Text)
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
        ),
    )
    source = Column(Text)
    copyright = Column(Text)
    reference = Column(Text)
    author_id = Column(UUID(as_uuid=True))
    publication_id = Column(UUID(as_uuid=True))
