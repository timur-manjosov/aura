"""Discord-facing command registration, grouped by knowledge-model area."""
from aura.commands.ask import register_ask_command
from aura.commands.config import register_config_command
from aura.commands.facts import register_fact_commands
from aura.commands.pending import register_pending_command
from aura.commands.proactive import register_proactive_commands
from aura.commands.supersede import register_supersede_command

__all__ = [
    "register_ask_command",
    "register_config_command",
    "register_fact_commands",
    "register_pending_command",
    "register_proactive_commands",
    "register_supersede_command",
]
