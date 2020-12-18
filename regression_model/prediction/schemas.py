from pydantic import BaseModel, Field
from enum import Enum


class SexEnum(str, Enum):
    """Enumeration for the value of the 'sex' input of the model."""

    male = "male"
    female = "female"


class RegionEnum(str, Enum):
    """Enumeration for the value of the 'region' input of the model."""

    southwest = "southwest"
    southeast = "southeast"
    northwest = "northwest"
    northeast = "northeast"


class InputSchema(BaseModel):
    """Schema for input of the model's predict method."""

    age: int = Field(None, description="Age of customer in years.")
    sex: SexEnum = Field(None, description="Sex of custumer.")
    bmi: float = Field(None, description="Body mass index of customer.")
    children: int = Field(None, description="Number of children of customer.")
    smoker: bool = Field(None, description="Whether customer is a smoker.")
    region: RegionEnum = Field(None, description="Region where customer lives.")


class OutputSchema(BaseModel):
    """Schema for output of the model's predict method."""

    charges: float = Field(None, description="Predicted charge to customer in USD.")
