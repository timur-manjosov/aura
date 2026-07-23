"""The knowledge model's data layer: schema, pydantic models, and data access."""
from aura.db.models import Fact, FactLink, FactStatus
from aura.db.repository import (
    CrossGuildLinkError,
    FactAlreadySupersededError,
    FactNotFoundError,
    RepositoryError,
    SelfLinkError,
    create_fact,
    get_active_facts,
    get_linked_facts,
    init_schema,
    link_facts,
    supersede_fact,
)

__all__ = [
    "CrossGuildLinkError",
    "Fact",
    "FactAlreadySupersededError",
    "FactLink",
    "FactNotFoundError",
    "FactStatus",
    "RepositoryError",
    "SelfLinkError",
    "create_fact",
    "get_active_facts",
    "get_linked_facts",
    "init_schema",
    "link_facts",
    "supersede_fact",
]
