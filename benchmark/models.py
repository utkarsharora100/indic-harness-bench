from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Language = Literal["english", "hindi", "hinglish"]


class Instructions(BaseModel):
    english: str
    hindi: str
    hinglish: str


class Environment(BaseModel):
    image: str
    workspace: str
    fixtures: str | None = None


class Limits(BaseModel):
    max_steps: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)


class Judge(BaseModel):
    command: str
    workdir: str = "."
    expected_exit_code: int = 0


class UpstreamTask(BaseModel):
    """The copied task assets from the reference Harness-Bench repository."""

    source_dir: str = "source"
    prompt_file: str = "prompt.txt"
    fixtures_dir: str = "fixtures"
    oracle_module: str = "oracle_grade.py"
    expected_outcome_score: float = Field(default=1.0, ge=0.0, le=1.0)


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    category: str
    source_task: str | None = None
    instruction: Instructions
    environment: Environment
    limits: Limits
    judge: Judge | None = None
    upstream: UpstreamTask | None = None

    @field_validator("task_id")
    @classmethod
    def task_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_id cannot be empty")
        return value

    @model_validator(mode="after")
    def requires_a_grader(self) -> TaskDefinition:
        if self.judge is None and self.upstream is None:
            raise ValueError("task must define either judge or upstream")
        return self

    def instruction_for(self, language: Language) -> str:
        return getattr(self.instruction, language)

    @classmethod
    def from_file(cls, path: Path) -> "TaskDefinition":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.model_validate(data)
