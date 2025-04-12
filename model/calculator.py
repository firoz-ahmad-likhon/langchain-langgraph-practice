from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input for multiplication tool."""

    a: int = Field(description="first number")
    b: int = Field(description="second number")
