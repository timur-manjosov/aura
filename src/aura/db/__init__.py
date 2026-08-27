"""The knowledge model's data layer: schema, pydantic models, and data access."""
from aura.db.models import Fact, FactLink, FactStatus
from aura.db.repository import (
    FactAlreadySupersededError,
    FactNotActiveError,
    FactNotFoundError,
    RepositoryError,
    SelfLinkError,
    create_fact,
    get_active_facts,
    get_linked_fact_ids,
    get_linked_facts,
    init_schema,
    link_facts,
    resolve_active_successors,
    supersede_fact,
    unlink_facts,
)

__all__ = [
    "Fact",
    "FactAlreadySupersededError",
    "FactLink",
    "FactNotActiveError",
    "FactNotFoundError",
    "FactStatus",
    "RepositoryError",
    "SelfLinkError",
    "create_fact",
    "get_active_facts",
    "get_linked_fact_ids",
    "get_linked_facts",
    "init_schema",
    "link_facts",
    "resolve_active_successors",
    "supersede_fact",
    "unlink_facts",
]
