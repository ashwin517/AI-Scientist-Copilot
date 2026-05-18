from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NumericSearchRange(BaseModel):
    min: float
    max: float
    steps: int = Field(default=5, ge=2, le=25)

    @model_validator(mode="after")
    def validate_bounds(self) -> "NumericSearchRange":
        if self.max <= self.min:
            raise ValueError("Search range max must be greater than min.")
        return self


class BatchReactorSearchSpace(BaseModel):
    temperature_c: NumericSearchRange = Field(
        default_factory=lambda: NumericSearchRange(min=70.0, max=110.0, steps=9)
    )
    batch_time_min: NumericSearchRange = Field(
        default_factory=lambda: NumericSearchRange(min=30.0, max=240.0, steps=8)
    )
    initial_concentration: NumericSearchRange = Field(
        default_factory=lambda: NumericSearchRange(min=0.5, max=2.0, steps=4)
    )
    catalyst_factor: NumericSearchRange = Field(
        default_factory=lambda: NumericSearchRange(min=0.5, max=2.0, steps=4)
    )


class BatchReactorOptimizationInput(BaseModel):
    penalty_weight: float = Field(default=1.0, ge=0.0)
    max_final_impurity: float | None = Field(default=0.15, ge=0.0)
    search_space: BatchReactorSearchSpace = Field(
        default_factory=BatchReactorSearchSpace
    )
    top_k: int = Field(default=5, ge=1, le=20)


class BatchReactorOptimizationCandidate(BaseModel):
    inputs: dict[str, float]
    final_yield: float
    final_impurity: float
    conversion: float
    objective_value: float
    constraint_satisfied: bool


class BatchReactorOptimizationResult(BaseModel):
    best_inputs: dict[str, float]
    best_final_yield: float
    best_final_impurity: float
    best_conversion: float
    objective_value: float
    top_candidates: list[BatchReactorOptimizationCandidate]
    evaluated_candidates: int
    feasible_candidates: int
    objective: str
    constraints: dict[str, float | None]
    search_space: dict[str, dict[str, float | int]]
    note: str


class OptimizationRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    optimization_type: str
    objective: str
    constraints_json: str
    search_space_json: str
    result_json: str
    created_at: datetime

