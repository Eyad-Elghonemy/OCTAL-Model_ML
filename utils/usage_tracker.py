"""
usage_tracker.py

Operation logger backed by Supabase (a managed external database),
so log entries survive container restarts/redeploys.

Resilience policy
-----------------
Logging is a side feature -- it must never be able to take the core
prediction endpoints down. Supabase's free tier pauses projects after
a period of inactivity, and while paused every request to it fails or
hangs. To keep the API healthy in that situation:

  * log_operation() is best-effort: it swallows any Supabase/network
    error (logging it to the server console instead) and never raises,
    so a prediction that succeeded is always returned to the caller.
  * A short request timeout stops a sleeping/unreachable database from
    stalling every prediction request for minutes.
  * The Supabase client is created lazily, so a missing/invalid config
    or an unreachable database can't crash the app at import time.
  * get_logs() -- the only place the database is genuinely required --
    raises LogsUnavailableError, which the /logs route turns into a
    clean 503 instead of an unhandled 500.
"""

import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from supabase import ClientOptions, create_client

logger = logging.getLogger(__name__)

CAIRO_TZ = ZoneInfo("Africa/Cairo")

_TABLE_NAME = "operation_log"

# Seconds to wait on Supabase before giving up. Short on purpose: a
# paused project can otherwise hang each request for a very long time.
_REQUEST_TIMEOUT_SECONDS = 5

_client = None


class LogsUnavailableError(Exception):
    """Raised when the operation log can't be read from the database."""


def _get_client():
    """Creates the Supabase client on first use and reuses it afterwards."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_KEY are not configured.")
        _client = create_client(
            url, key,
            options=ClientOptions(postgrest_client_timeout=_REQUEST_TIMEOUT_SECONDS),
        )
    return _client


def log_operation(operation_type: str, success: bool) -> bool:
    """
    Records one API operation (e.g. 'linear', 'from-vin') on a
    best-effort basis.

    Returns:
        True if the entry was stored, False if the database was
        unreachable. Never raises -- a logging failure must not
        affect the request that triggered it.
    """
    try:
        _get_client().table(_TABLE_NAME).insert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation_type": operation_type,
            "success": success,
        }).execute()
        return True
    except Exception as e:
        logger.warning("Could not write to operation log (Supabase unavailable?): %s", e)
        return False


def get_logs() -> list:
    """
    Returns all logged operations, most recent first, with the
    timestamp reformatted into a display-ready string
    (e.g. '19/07/2026 11:16 PM').

    Raises:
        LogsUnavailableError: if the database can't be reached.
    """
    try:
        response = (
            _get_client().table(_TABLE_NAME)
            .select("timestamp, operation_type, success")
            .order("id", desc=True)
            .execute()
        )
    except Exception as e:
        logger.warning("Could not read operation log (Supabase unavailable?): %s", e)
        raise LogsUnavailableError from e

    formatted = []
    for e in response.data:
        try:
            dt = datetime.fromisoformat(e["timestamp"]).astimezone(CAIRO_TZ)
            display_time = dt.strftime("%d/%m/%Y %I:%M %p")
        except (KeyError, ValueError, TypeError):
            display_time = e.get("timestamp", "")
        formatted.append({
            "timestamp": display_time,
            "operation_type": e.get("operation_type"),
            "success": e.get("success"),
        })
    return formatted
