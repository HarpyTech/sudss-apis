from datetime import datetime, timezone

def get_timestamp() -> str:
    """Return current UTC timestamp as an ISO-8601 string with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
