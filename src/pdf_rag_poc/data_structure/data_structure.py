from dataclasses import dataclass, field
import numpy as np
import uuid

@dataclass
class Chunk:
    text: str
    id: int
    embedding: np.ndarray

@dataclass
class Document: 
    tenant_id: str
    file_name: str
    text: str
    chunks: list[Chunk]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class QandA:
    question: str
    answer: list[str]