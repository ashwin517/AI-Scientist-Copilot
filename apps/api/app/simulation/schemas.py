from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class BatchReactorSimulationInput(BaseModel):
    temperature: float = Field(
        default=85.0,
        gt=-273.15,
        description="Batch temperature in degrees Celsius.",
    )
    batch_time: float = Field(
        default=120.0,
        gt=0,
        description="Batch duration in minutes.",
    )
    initial_concentration: float = Field(
        default=1.0,
        gt=0,
        description="Initial A concentration in arbitrary concentration units.",
    )
    catalyst_factor: float = Field(
        default=1.0,
        ge=0,
        description="Simple multiplier applied to both kinetic rates.",
    )
    time_points: int = Field(default=121, ge=2, le=1000)


class BatchReactorSimulationOutput(BaseModel):
    time_grid: list[float]
    CA_profile: list[float]
    CB_profile: list[float]
    CC_profile: list[float]
    final_yield: float
    final_impurity: float
    conversion: float
    rate_constants: dict[str, float]
    note: str


class SimulationRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    simulation_type: str
    input_json: str
    result_json: str
    created_at: datetime
