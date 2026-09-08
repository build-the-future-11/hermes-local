from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str
    content: str
    finish_reason: Literal["stop", "length"]
    prompt_characters: int = Field(ge=0)
    generated_characters: int = Field(ge=0)
    evidence: dict[str, Any]
