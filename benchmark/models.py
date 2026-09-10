from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    category: str
    source_task: str | None = None
    instruction: Instructions
    environment: Environment
    limits: Limits
    judge: Judge

    @field_validator("task_id")
    @classmethod
    def task_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_id cannot be empty")
        return value

    def instruction_for(self, language: Language) -> str:
        return getattr(self.instruction, language)

    @classmethod
    def from_file(cls, path: Path) -> "TaskDefinition":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.model_validate(data)
