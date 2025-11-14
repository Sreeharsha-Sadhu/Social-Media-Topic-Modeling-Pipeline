"""
Live LinkedIn fetcher (simulated / lightweight).

Signature matches fetcher_callable expected by Stage 4:
    fetch_linkedin_user_posts(user_id: str, since: Optional[datetime], limit: int) -> List[Dict]

LinkedIn's official APIs are restricted; this module:
 - uses service token if present (not implemented here),
 - otherwise simulates deterministic posts.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from dateutil import parser as date_parser

from src.core import config
from src.core.logging_config import get_logger
from src.core import utils

logger = get_logger(__name__)

SEED = 24
random.seed(SEED)


def _ensure_aware(dt):
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = date_parser.isoparse(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _simulate_linkedin_fetch(user_id: str, since: Optional[datetime], limit: int) -> List[Dict]:
    """
    Simulate LinkedIn posts for a user (professional tone).
    """
    if isinstance(since, str):
        since = _ensure_aware(since)

    results = []
    for i in range(limit):
        created_at = datetime.now(timezone.utc) - timedelta(hours=i * 12 + random.randint(0, 8))
        if since and created_at <= since:
            break
        text = (
            f"Professional insight on data science and product work #{random.randint(1, 200)}. "
            f"Reflections on team processes and scaling."
        )
        results.append({
            "post_id": f"sim_linkedin_{int(created_at.timestamp())}_{i}",
            "text": text,
            "created_at": created_at.isoformat()
        })
    return sorted(results, key=lambda p: p["created_at"], reverse=True)


def fetch_linkedin_user_posts(user_id: str, since: Optional[datetime] = None, limit: int = 100) -> List[Dict]:
    """
    Public fetcher used by the live pipeline.

    For now it uses simulation. If you later add a token-based LinkedIn fetch,
    place it here and preserve return schema.
    """
    logger.info("Fetching LinkedIn posts for %s (limit=%s, since=%s)", user_id, limit, since)
    try:
        since_aware = _ensure_aware(since) if since else None
        posts = _simulate_linkedin_fetch(user_id, since_aware, limit)
        logger.info("Fetched %d LinkedIn (simulated) posts for %s", len(posts), user_id)
        return posts
    except Exception as e:
        logger.error("LinkedIn fetcher error for %s: %s", user_id, e)
        return []
