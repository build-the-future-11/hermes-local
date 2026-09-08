from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
    )


class HypothesisStatus(StrEnum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    PARKED = "parked"


class Evidence(StrictModel):
    source: str
    detail: str
    weight: float = Field(ge=0.0, le=1.0, default=0.5)
    tags: list[str] = Field(default_factory=list)


class Hypothesis(StrictModel):
    name: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    evidence: list[Evidence] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)


class ProvenanceRecord(StrictModel):
    actor: str
    action: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComputeBudget(StrictModel):
    token_budget: int = Field(ge=0, default=2_000)
    model_call_budget: int = Field(ge=0, default=8)
    branch_budget: int = Field(ge=1, default=6)
    tool_budget: int = Field(ge=0, default=6)
    latency_budget_ms: int = Field(ge=0, default=1_500)
    monetary_budget_usd: float = Field(ge=0.0, default=0.0)
    memory_budget_mb: int = Field(ge=1, default=512)
    depth_budget: int = Field(ge=1, default=4)


class BranchScore(StrictModel):
    plausibility: float = Field(ge=0.0, le=1.0)
    information_gain: float = Field(ge=0.0, le=1.0)
    usefulness: float = Field(ge=0.0, le=1.0)
    uncertainty_value: float = Field(ge=0.0, le=1.0)
    compute_cost: float = Field(gt=0.0, default=1.0)

    def priority(self) -> float:
        numerator = (
            self.plausibility * self.information_gain * self.usefulness * self.uncertainty_value
        )
        return numerator / self.compute_cost
