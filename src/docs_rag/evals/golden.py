"""Golden set schema. `data/golden/golden.schema.json` is generated from this model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal["lookup", "how_to", "keyword", "paraphrase", "multi_hop", "unanswerable"]


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q\d{3}$")
    question: str = Field(min_length=8)
    type: QuestionType
    expected_pages: list[str] = Field(description="page ids that contain the answer; empty if unanswerable")
    expected_sections: list[str] = Field(default_factory=list, description="headings where the evidence is")
    reference_answer: str
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> GoldenItem:
        if self.type == "unanswerable" and self.expected_pages:
            raise ValueError(f"{self.id}: unanswerable items must have no expected_pages")
        if self.type != "unanswerable" and not self.expected_pages:
            raise ValueError(f"{self.id}: answerable items need expected_pages")
        if self.type == "multi_hop" and len(set(self.expected_pages)) != 2:
            raise ValueError(f"{self.id}: multi_hop items need exactly two distinct pages")
        return self

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_golden(path: Path) -> list[GoldenItem]:
    items = [GoldenItem.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate golden ids")
    return items


def json_schema() -> str:
    return json.dumps(GoldenItem.model_json_schema(), indent=2) + "\n"


def in_corpus(items: Iterable[GoldenItem], page_ids: set[str]) -> list[GoldenItem]:
    """Items evaluable on a corpus: all expected pages present (unanswerables always kept)."""
    return [i for i in items if set(i.expected_pages) <= page_ids]
