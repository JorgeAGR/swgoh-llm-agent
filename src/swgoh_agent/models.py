"""Domain models for SWGOH units and their abilities."""

from typing import Literal

from pydantic import BaseModel, Field

CombatType = Literal["character", "ship"]


class Ability(BaseModel):
    base_id: str
    name: str
    kind: str  # Basic, Special, Leader, Unique, Ultimate, Crew, Hardware, Contract
    description: str
    is_zeta: bool = False
    is_omicron: bool = False
    omicron_modes: list[str] = Field(default_factory=list)

    @property
    def upgrade_labels(self) -> list[str]:
        labels = []
        if self.is_zeta:
            labels.append("Zeta")
        if self.is_omicron:
            modes = ", ".join(self.omicron_modes)
            labels.append(f"Omicron: {modes}" if modes else "Omicron")
        return labels

    def render(self) -> str:
        labels = "".join(f" [{label}]" for label in self.upgrade_labels)
        return f"{self.name} ({self.kind}){labels}\n{self.description}"


class Unit(BaseModel):
    base_id: str
    name: str
    url: str
    combat_type: CombatType
    alignment: str
    role: str
    description: str = ""
    categories: list[str] = Field(default_factory=list)
    ability_classes: list[str] = Field(default_factory=list)
    abilities: list[Ability] = Field(default_factory=list)
    # Characters: the ship they pilot. Ships: the characters that crew them.
    ship: str | None = None
    crew: list[str] = Field(default_factory=list)
    is_capital_ship: bool = False
    base_stats: dict[str, str] | None = None

    @property
    def is_galactic_legend(self) -> bool:
        return "Galactic Legend" in self.categories

    @property
    def is_leader(self) -> bool:
        return "Leader" in self.categories

    def summary_line(self) -> str:
        return f"{self.name} — {self.role}, {', '.join(self.categories)}"


class Dataset(BaseModel):
    fetched_at: str
    source: str
    units: list[Unit]
