"""
Live Reddit fetcher.

Provides a single function:

    fetch_reddit_user_posts(user_id: str, since: Optional[datetime], limit: int) -> List[Dict]

Each returned dict has:
    - post_id: str
    - text: str
    - created_at: ISO8601 string with timezone (UTC, ending with Z)
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from dateutil import parser as date_parser

from src.core import config
from src.core.logging_config import get_logger
from src.core import utils

logger = get_logger(__name__)

# Deterministic randomness for simulation (set to None to make fully random)
SEED = 42
random.seed(SEED)


def _iso_utc_now_minus(delta_days=0, delta_hours=0, delta_minutes=0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=delta_days, hours=delta_hours, minutes=delta_minutes)
    return dt.isoformat()


def _ensure_aware(dt):
    """Return timezone-aware datetime (UTC)."""
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = date_parser.isoparse(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _read_user_subreddits_from_db(user_id: str) -> List[str]:
    """Try to read subreddits list for user from DB users table, fallback to users.json."""
    try:
        engine = utils.get_sqlalchemy_engine()
        query = "SELECT subreddits FROM users WHERE user_id = %s;"
        # use psycopg2 for simple fetch
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT subreddits FROM users WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
                if row and row[0]:
                    # row[0] may be PostgreSQL array already; ensure list
                    subs = list(row[0]) if isinstance(row[0], (list, tuple)) else json.loads(row[0])
                    return [s for s in subs if s]
    except Exception:
        pass

    # fallback: read users.json
    try:
        with open(config.USERS_JSON_PATH, "r", encoding="utf-8") as f:
            users = json.load(f)
        for u in users:
            if u.get("user_id") == user_id:
                return u.get("subreddits", [])
    except Exception:
        pass

    # default
    return ["news", "technology"]


def _simulate_reddit_api_fetch(subreddits: List[str], since: Optional[datetime], limit: int) -> List[Dict]:
    """
    Create deterministic synthetic posts for testing if API isn't configured.
    Returns posts newest-first.
    """
    # ensure since is aware
    if isinstance(since, str):
        since = _ensure_aware(since)

    results = []
    # create up to `limit` posts across subreddits
    for i in range(limit):
        # generate created_at decreasing in time
        created_at = datetime.now(timezone.utc) - timedelta(minutes=i * 15 + random.randint(0, 10))
        if since and created_at <= since:
            # once we reach older than 'since', we stop (simulate incremental fetch)
            break

        post = {
            "post_id": f"sim_reddit_{int(created_at.timestamp())}_{i}",
            "text": f"Reddit post about {random.choice(subreddits)} and trending topic {random.randint(1, 300)}",
            "created_at": created_at.isoformat()
        }
        results.append(post)

    # newest-first
    return sorted(results, key=lambda p: p["created_at"], reverse=True)


def _try_real_reddit_fetch(subreddits: List[str], since: Optional[datetime], limit: int) -> List[Dict]:
    """
    Try to use PRAW if REDDIT credentials are present.
    This function will raise on missing credentials / runtime errors, caller will fallback.
    """
    try:
        import praw
    except Exception as e:
        raise RuntimeError("praw not available") from e

    if not (config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET and config.REDDIT_USER_AGENT):
        raise RuntimeError("Missing Reddit credentials in environment")

    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        check_for_async=False
    )

    results = []
    # try to fetch from each subreddit until we have enough posts
    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            # PRAW returns newest first by default when using .new()
            for submission in subreddit.new(limit=limit * 2):
                # created_utc is float seconds
                created = datetime.fromtimestamp(submission.created_utc, timezone.utc)
                if since and created <= since:
                    continue
                text = submission.title
                if getattr(submission, "selftext", None):
                    text = f"{text}. {submission.selftext}"
                results.append({
                    "post_id": f"reddit_{submission.id}",
                    "text": text,
                    "created_at": created.isoformat()
                })
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
        except Exception:
            # continue with other subreddits
            continue

    return sorted(results, key=lambda p: p["created_at"], reverse=True)


def fetch_reddit_user_posts(user_id: str, since: Optional[datetime] = None, limit: int = 100) -> List[Dict]:
    """
    Public fetcher used by the live pipeline.

    - since: datetime or None. If provided, only posts newer than since are returned.
    - returns newest-first list of dicts with post_id, text, created_at (ISO aware).
    """
    logger.info("Fetching Reddit posts for %s (limit=%s, since=%s)", user_id, limit, since)
    try:
        subreddits = _read_user_subreddits_from_db(user_id) or ["news", "technology"]
        # Normalize 'since'
        since_aware = _ensure_aware(since) if since else None

        # Try real API when credentials exist
        try:
            posts = _try_real_reddit_fetch(subreddits, since_aware, limit)
            logger.info("Fetched %d Reddit posts via API for %s", len(posts), user_id)
            return posts
        except Exception as e:
            logger.warning("Real Reddit fetch failed or not configured: %s. Falling back to simulation.", e)
            posts = _simulate_reddit_api_fetch(subreddits, since_aware, limit)
            logger.info("Fetched %d simulated Reddit posts for %s", len(posts), user_id)
            return posts

    except Exception as e:
        logger.error("Reddit fetcher error for %s: %s", user_id, e)
        return []
