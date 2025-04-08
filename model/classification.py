from pydantic import BaseModel, Field
from typing import Literal


class Classification(BaseModel):
    """Classification of a statement."""

    sentiment: Literal["happy", "neutral", "sad"]
    aggressiveness: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="Describes how aggressive the statement is; the higher the number, the more aggressive.",
    )
    language: Literal["spanish", "english", "french", "german", "italian"]
