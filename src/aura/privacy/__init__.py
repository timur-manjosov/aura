"""Data obligations (P7a): the deletion ledger, executing deletion requests, the purge job,
the author lookup and the knowledge-base export.

Everything here acts on the knowledge model; nothing here is a fifth trigger.
The deletion rules themselves live in `aura.db.deletion`.
"""
