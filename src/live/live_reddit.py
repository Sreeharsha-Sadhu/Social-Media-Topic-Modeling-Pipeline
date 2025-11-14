"""
live_reddit.py

Provides:
 - fetch_reddit_user_posts(user_id, since, limit)
 - analyze_reddit_feed(user_id, top_n=50, prefer_since_last_run=True)

Behavior:
 - If PRAW credentials are present in src.core.config, attempt to use PRAW.
 - Otherwise use a safe simulator that produces deterministic example posts.
 - Normalizes timestamps to timezone-aware UTC datetimes.
 - Persists fetched posts into `posts` table (INSERT ... ON CONFLICT DO NOTHING)
   so later post_topic_mapping inserts won't violate foreign keys.
 - Calls stage4.run_live_analysis_for_user(...) as the orchestrator.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

try:
    import praw
except Exception:
    praw = None

from src.core import config, utils
from src.core.logging_config import get_logger
from src.data_pipeline.stage4_global_analysis import run_live_analysis_for_user

logger = get_logger(__name__)


# -------------------------
# Helpers
# -------------------------
def _to_utc_iso(ts_str_or_dt) -> str:
    """
    Normalize a timestamp (isostring or datetime) to an ISO8601 string with UTC offset.
    Returns string suitable for DB insertion and comparison.
    """
    if isinstance(ts_str_or_dt, str):
        try:
            # fromisoformat handles offset if present
            dt = datetime.fromisoformat(ts_str_or_dt)
        except Exception:
            # fallback: parse naive formats
            try:
                dt = datetime.strptime(ts_str_or_dt, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                dt = datetime.utcnow()
    elif isinstance(ts_str_or_dt, datetime):
        dt = ts_str_or_dt
    else:
        dt = datetime.utcnow()

    # if naive, set tzinfo=UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _persist_posts_to_db(posts: List[Dict[str, str]]):
    """
    Insert posts into the `posts` table if they don't already exist.
    Uses ON CONFLICT DO NOTHING semantics via psycopg2 execute.
    Expects each post to have keys: post_id, author_id (optional), content, created_at (ISO).
    """
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
                    created_at = _to_utc_iso(p.get("created_at"))  # normalize
                    # lightweight cleaning (Stage 3 will do full cleaning when ETL runs)
                    cleaned = (
                        content.lower()
                        .replace("\n", " ")
                    )
                    cur.execute(insert_sql, (post_id, author_id, content, cleaned, created_at))
                conn.commit()
    except Exception as e:
        logger.warning("Could not persist fetched posts to DB (non-fatal): %s", e)


# -------------------------
# Reddit fetcher
# -------------------------
def _simulate_reddit_api_fetch(subreddits: List[str], since, limit: int = 50) -> List[Dict]:
    """
    Deterministic simulator that returns a list of posts.
    Generates stable post_ids using a pseudo-random generator seeded by time/subreddit.
    """
    out = []
    # create a deterministic seed to make outputs reproducible per run
    seed = int(time.time()) % (2**31 - 1)
    rnd = random.Random(seed)
    now = datetime.now(timezone.utc)
    for i in range(limit):
        # create ISO timestamp spaced by minutes backwards
        created = (now.replace(microsecond=0) - (i * utils.timedelta(minutes=1))) if hasattr(utils, "timedelta") else (now - utils.timedelta(minutes=i)) if hasattr(utils, "timedelta") else (now)
        # ensure since filtering: if since provided and newer than created skip
        if since:
            try:
                # since might be datetime; normalize both to aware UTC
                if isinstance(since, str):
                    since_dt = datetime.fromisoformat(since)
                else:
                    since_dt = since
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=timezone.utc)
                if created <= since_dt:
                    # skip older posts
                    continue
            except Exception:
                pass

        post_id = f"reddit_{rnd.randint(10, 10**8)}"
        author = f"reddit_user_{rnd.randint(1, 200)}"
        content = f"Reddit post about {rnd.choice(['python','react','node','pytorch','webdev','nba'])} and trending topic {rnd.randint(1,500)}"
        out.append({
            "post_id": post_id,
            "author_id": author,
            "content": content,
            "created_at": created.isoformat()
        })
    return out


def fetch_reddit_user_posts(user_id: str, since: Optional[datetime] = None, limit: int = 50) -> List[Dict]:
    """
    Fetch recent Reddit posts for the user's subscribed subreddits.
    - Attempts to use PRAW if credentials are present and praw is installed.
    - If any issue, falls back to simulation.
    The returned list contains dicts: post_id, author_id, content/text, created_at(ISO)
    """
    logger.info("[live_reddit] Fetching Reddit posts for %s (limit=%s, since=%s)", user_id, limit, since)
    # If real PRAW usage desired
    if praw and config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET and config.REDDIT_USER_AGENT:
        try:
            reddit = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT,
            )
            # find user's subreddits from users table
            subs = []
            try:
                with utils.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT subreddits FROM users WHERE user_id=%s", (user_id,))
                        row = cur.fetchone()
                        if row and row[0]:
                            subs = row[0]
            except Exception:
                subs = []

            if not subs:
                # fallback to some default subs
                subs = ["python", "webdev", "MachineLearning"]

            results = []
            fetched = 0
            for sub in subs:
                if fetched >= limit:
                    break
                subreddit = reddit.subreddit(sub)
                for submission in subreddit.new(limit=limit):
                    # convert created_utc (float) to aware datetime
                    created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
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
                    pid = f"reddit_{submission.id}"
                    results.append({
                        "post_id": pid,
                        "author_id": getattr(submission, "author", None).name if getattr(submission, "author", None) else None,
                        "content": submission.title + "\n\n" + (submission.selftext or ""),
                        "created_at": created.isoformat()
                    })
                    fetched += 1
                    if fetched >= limit:
                        break
            logger.info("[live_reddit] Fetched %d Reddit posts for %s", len(results), user_id)
            # persist posts into posts table (idempotent)
            _persist_posts_to_db(results)
            return results
        except Exception as e:
            logger.warning("[live_reddit] PRAW fetch failed or misconfigured: %s. Falling back to simulator.", e)

    # simulator fallback
    try:
        # since may be datetime or iso string; convert to aware datetime for simulator
        s = since
        if s and isinstance(s, str):
            try:
                s = datetime.fromisoformat(s)
            except Exception:
                s = None
        if s and s.tzinfo is None:
            s = s.replace(tzinfo=timezone.utc)
        simulated = _simulate_reddit_api_fetch([], s, limit=limit)
        _persist_posts_to_db(simulated)
        logger.info("[live_reddit] Fetched %d Reddit posts for %s (simulated)", len(simulated), user_id)
        return simulated
    except Exception as e:
        logger.exception("[live_reddit] Simulator failed: %s", e)
        return []


# -------------------------
# High-level analyzer invoked from UI/CLI
# -------------------------
def analyze_reddit_feed(user_id: str, top_n: int = 50, prefer_since_last_run: bool = True):
    """
    Convenience wrapper used by the UI/CLI.
    Calls run_live_analysis_for_user with this module's fetcher.
    Prints a short user-friendly summary after completion.
    """
    print("Running live Reddit analysis...")
    result = run_live_analysis_for_user(
        user_id=user_id,
        source="reddit",
        fetcher_callable=fetch_reddit_user_posts,
        top_n=top_n,
        prefer_since_last_run=prefer_since_last_run,
    )

    # pretty-print result
    status = result.get("status")
    payload = result.get("payload") or {}
    print("\nLive Analysis Result:")
    print(f"Status: {status}")
    print(f"User: {user_id}")
    print(f"Source: reddit")
    if payload:
        print(f"Posts processed: {payload.get('post_count')}")
        print(f"Run at: {payload.get('created_at')}")
        # print titles + short summary
        res = payload.get("result", {})
        topics = res.get("topics", [])
        if topics:
            for t in topics:
                print("\n" + "-" * 60)
                title = t.get("title") or t.get("topic_title") or "Untitled"
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
