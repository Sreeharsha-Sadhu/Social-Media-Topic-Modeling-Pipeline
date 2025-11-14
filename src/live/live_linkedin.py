"""
live_linkedin.py

Provides:
 - fetch_linkedin_user_posts(user_id, since, limit)
 - analyze_linkedin_feed(user_id, top_n=50, prefer_since_last_run=True)

Behavior:
 - LinkedIn does not have a public simple client in this repo; so this module
   attempts to use a hypothetical API client if provided via config, otherwise
   uses a simulator similar to the Reddit simulator.
 - Persists posts to `posts` table (ON CONFLICT DO NOTHING).
 - Calls stage4.run_live_analysis_for_user(...) for orchestration.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

from src.core import config, utils
from src.core.logging_config import get_logger
from src.data_pipeline.stage4_global_analysis import run_live_analysis_for_user

logger = get_logger(__name__)


# -------------------------
# Helpers
# -------------------------
def _to_utc_iso(ts_str_or_dt) -> str:
    if isinstance(ts_str_or_dt, str):
        try:
            dt = datetime.fromisoformat(ts_str_or_dt)
        except Exception:
            dt = datetime.utcnow()
    elif isinstance(ts_str_or_dt, datetime):
        dt = ts_str_or_dt
    else:
        dt = datetime.utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _persist_posts_to_db(posts: List[Dict[str, str]]):
    if not posts:
        return
    insert_sql = """
        INSERT INTO posts (post_id, author_id, content, cleaned_content, created_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (post_id) DO NOTHING
    """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                for p in posts:
                    post_id = p.get("post_id")
                    author_id = p.get("author_id")
                    content = p.get("content") or p.get("text") or ""
                    created_at = _to_utc_iso(p.get("created_at"))
                    cleaned = content.lower().replace("\n", " ")
                    cur.execute(insert_sql, (post_id, author_id, content, cleaned, created_at))
                conn.commit()
    except Exception as e:
        logger.warning("Could not persist LinkedIn posts to DB (non-fatal): %s", e)


# -------------------------
# LinkedIn fetcher (simulator + optional real client)
# -------------------------
def _simulate_linkedin_fetch(user_id: str, since, limit: int = 50) -> List[Dict]:
    out = []
    rnd = random.Random(int(time.time()) % (2**31 - 1))
    now = datetime.now(timezone.utc)
    for i in range(limit):
        created = now - utils.timedelta(minutes=i) if hasattr(utils, "timedelta") else now
        if since:
            try:
                s = since
                if isinstance(s, str):
                    s = datetime.fromisoformat(s)
                if s.tzinfo is None:
                    s = s.replace(tzinfo=timezone.utc)
                if created <= s:
                    continue
            except Exception:
                pass
        pid = f"linkedin_{rnd.randint(10, 10**8)}"
        author = f"linkedin_user_{rnd.randint(1,300)}"
        content = f"LinkedIn post about career, data science, and industry trends {rnd.randint(1,999)}"
        out.append({
            "post_id": pid,
            "author_id": author,
            "content": content,
            "created_at": created.isoformat()
        })
    return out


def fetch_linkedin_user_posts(user_id: str, since: Optional[datetime] = None, limit: int = 50) -> List[Dict]:
    """
    Attempt to fetch LinkedIn posts for the user. If a real client is not configured,
    uses a simulator. Always persists posts into DB (idempotent).
    """
    logger.info("[live_linkedin] Fetching LinkedIn posts for %s (limit=%d, since=%s)", user_id, limit, since)
    # If you later add a real LinkedIn client, attempt it here (guarded).
    try:
        # simulator path
        s = since
        if s and isinstance(s, str):
            try:
                s = datetime.fromisoformat(s)
            except Exception:
                s = None
        if s and s.tzinfo is None:
            s = s.replace(tzinfo=timezone.utc)
        posts = _simulate_linkedin_fetch(user_id, s, limit=limit)
        _persist_posts_to_db(posts)
        logger.info("[live_linkedin] Simulated fetch %d posts for %s", len(posts), user_id)
        return posts
    except Exception as e:
        logger.exception("[live_linkedin] Fetch failed: %s", e)
        return []


# -------------------------
# High-level analyzer
# -------------------------
def analyze_linkedin_feed(user_id: str, top_n: int = 50, prefer_since_last_run: bool = True):
    print("Running live LinkedIn analysis...")
    result = run_live_analysis_for_user(
        user_id=user_id,
        source="linkedin",
        fetcher_callable=fetch_linkedin_user_posts,
        top_n=top_n,
        prefer_since_last_run=prefer_since_last_run,
    )

    # pretty-print
    status = result.get("status")
    payload = result.get("payload") or {}
    print("\nLive Analysis Result:")
    print(f"Status: {status}")
    print(f"User: {user_id}")
    print(f"Source: linkedin")
    if payload:
        print(f"Posts processed: {payload.get('post_count')}")
        print(f"Run at: {payload.get('created_at')}")
        res = payload.get("result", {})
        topics = res.get("topics", [])
        if topics:
            for t in topics:
                print("\n" + "-" * 60)
                title = t.get("title") or "Untitled"
                summary = t.get("summary") or ""
                bullets = t.get("bullets") or []
                sentiment = t.get("sentiment") or {}
                print(f"Topic: {title}")
                print(f"Sentiment: {sentiment.get('label')} ({sentiment.get('score')})")
                print("\nSummary:\n" + summary[:1000])
                if bullets:
                    print("\nKey points:")
                    for b in bullets:
                        print(" - " + b)
        else:
            print("No topics generated.")
    else:
        print("No payload returned.")
    return result
