"""
live_reddit.py
──────────────────────────────────────────────────────
Reddit-first live fetcher.

Primary source: PRAW (config via REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT)
Fallback: deterministic simulator that produces posts in the exact Reddit schema.

Responsibilities:
 - Fetch per-subreddit top posts (adaptive allocation)
 - Auto-create stub users when a Reddit author is unknown in users table
 - Persist posts to `posts` table in full Reddit schema with ON CONFLICT DO NOTHING
 - Return structured results for downstream analysis
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

try:
    import praw
except Exception:
    praw = None

from src.core import config, utils
from src.core.logging_config import get_logger
from psycopg2.extras import execute_batch

logger = get_logger(__name__)

# -------------------------
# Helpers
# -------------------------
def _to_utc_iso(ts) -> str:
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                dt = datetime.utcnow()
    else:
        dt = datetime.utcnow()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


# -------------------------
# Auto-create user stub (prevents FK failures)
# -------------------------
def _ensure_author_exists(author_id: Optional[str]) -> None:
    if not author_id:
        return
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (user_id, username, personas, subreddits) VALUES (%s,%s,%s,%s) "
                    "ON CONFLICT (user_id) DO NOTHING;",
                    (author_id, author_id, [], [])
                )
                conn.commit()
    except Exception as e:
        logger.debug("[live_reddit] _ensure_author_exists failed for %s: %s", author_id, e)


# -------------------------
# Persist posts to DB (full Reddit schema)
# -------------------------
def _persist_posts_to_db(posts: List[Dict[str, Any]]):
    if not posts:
        return

    sql = """
        INSERT INTO posts (
            post_id, author_id, subreddit, title, selftext, content,
            cleaned_content, created_at, score, num_comments, flair
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (post_id) DO NOTHING;
    """

    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                batch = []
                for p in posts:
                    # ensure author exists to satisfy FK
                    _ensure_author_exists(p.get("author_id"))

                    title = p.get("title") or ""
                    body = p.get("selftext") or ""
                    content = p.get("content") or f"{title}\n\n{body}"
                    cleaned = content.lower().replace("\n", " ")
                    created = _to_utc_iso(p.get("created_at"))

                    batch.append((
                        p.get("post_id"),
                        p.get("author_id"),
                        p.get("subreddit"),
                        title,
                        body,
                        content,
                        cleaned,
                        created,
                        p.get("score"),
                        p.get("num_comments"),
                        p.get("flair")
                    ))

                execute_batch(cur, sql, batch)
                conn.commit()
        logger.info("[live_reddit] Persisted %d posts to DB.", len(posts))
    except Exception as e:
        logger.exception("[live_reddit] Could not persist posts to DB: %s", e)


# -------------------------
# Simulator (deterministic-like)
# -------------------------
SIM_TOPICS = {
    "machinelearning": ["transformers", "scaling", "fine-tuning"],
    "datascience": ["eda", "feature engineering", "visualization"],
    "python": ["asyncio", "typing", "pandas"],
    "technology": ["cloud", "hardware", "ai"],
    "news": ["policy", "elections", "global"],
}

def _simulate_posts_for_subreddit(subreddit: str, limit: int = 20, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    seed = abs(hash(subreddit)) ^ int(time.time() % 100000)
    rnd = random.Random(seed)
    now = datetime.now(timezone.utc)
    topics = SIM_TOPICS.get(subreddit.lower(), ["discussion", "news"])

    out = []
    for i in range(limit):
        topic = rnd.choice(topics)
        title = f"{topic.capitalize()} update in r/{subreddit}"
        body = f"Quick note about {topic}. Example {rnd.randint(1,9999)}."
        created = now - timedelta(minutes=rnd.randint(0, 60) + i)
        if since and created <= since:
            continue
        out.append({
            "post_id": f"sim_{subreddit}_{rnd.randint(1,10**12)}",
            "author_id": f"sim_user_{rnd.randint(1,9999)}",
            "subreddit": subreddit,
            "title": title,
            "selftext": body,
            "content": f"{title}\n\n{body}",
            "created_at": created.isoformat(),
            "score": rnd.randint(0, 1000),
            "num_comments": rnd.randint(0, 200),
            "flair": None
        })
    return out


# -------------------------
# PRAW fetch for a subreddit
# -------------------------
def _fetch_via_praw_for_subreddit(subreddit: str, limit: int = 50, sort: str = "top", since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    if praw is None:
        raise RuntimeError("PRAW not installed")

    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )
    out = []
    try:
        sub_obj = reddit.subreddit(subreddit)
        iterator = sub_obj.top(limit=limit) if sort == "top" else sub_obj.new(limit=limit)
    except Exception as e:
        logger.warning("[live_reddit] PRAW cannot access %s: %s", subreddit, e)
        return out

    for submission in iterator:
        created = datetime.fromtimestamp(getattr(submission, "created_utc", time.time()), tz=timezone.utc)
        if since and created <= since:
            continue
        title = getattr(submission, "title", "") or ""
        selftext = getattr(submission, "selftext", "") or ""
        out.append({
            "post_id": f"reddit_{getattr(submission, 'id', str(time.time()))}",
            "author_id": getattr(submission.author, "name", None) if getattr(submission, "author", None) else None,
            "subreddit": subreddit,
            "title": title,
            "selftext": selftext,
            "content": f"{title}\n\n{selftext}",
            "created_at": created.isoformat(),
            "score": getattr(submission, "score", None),
            "num_comments": getattr(submission, "num_comments", None),
            "flair": getattr(submission, "link_flair_text", None)
        })
    return out


# -------------------------
# Activity estimation helper (same idea as before)
# -------------------------
def _estimate_activity_for_subreddit(subreddit: str, sample_limit: int = 12) -> float:
    try:
        if praw and config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET and config.REDDIT_USER_AGENT:
            sample = _fetch_via_praw_for_subreddit(subreddit, limit=sample_limit, sort="new")
        else:
            sample = _simulate_posts_for_subreddit(subreddit, limit=min(sample_limit, 6))
    except Exception:
        sample = _simulate_posts_for_subreddit(subreddit, limit=min(sample_limit, 6))

    if not sample:
        return 1.0
    scores = [max(0, (p.get("score") or 0)) for p in sample]
    comments = [max(0, (p.get("num_comments") or 0)) for p in sample]
    now = datetime.now(timezone.utc)
    recency = []
    for p in sample:
        try:
            created = datetime.fromisoformat(p.get("created_at"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            delta = (now - created).total_seconds()
            recency.append(max(0.0, 1.0 / (1.0 + delta / (60*60*24))))
        except Exception:
            recency.append(0.1)
    avg_score = sum(scores) / len(scores)
    avg_comments = sum(comments) / len(comments)
    avg_recency = sum(recency) / len(recency)
    alpha, beta, gamma = 0.5, 0.3, 0.2
    activity = alpha * avg_score + beta * avg_comments + gamma * (avg_recency * 100.0)
    return float(max(0.1, activity))


# -------------------------
# Allocation (same algorithm)
# -------------------------
def _allocate_posts_across_subreddits(activity_map: Dict[str, float], top_n_total: int,
                                      min_per_sub: int = 5, max_per_sub: int = 40) -> Dict[str, int]:
    subs = list(activity_map.keys())
    if not subs:
        return {}
    raw_scores = {s: max(0.0001, float(activity_map[s])) for s in subs}
    total_score = sum(raw_scores.values())
    alloc = {s: int(round((raw_scores[s] / total_score) * top_n_total)) for s in subs}
    for s in subs:
        if alloc[s] < min_per_sub:
            alloc[s] = min_per_sub
        if alloc[s] > max_per_sub:
            alloc[s] = max_per_sub
    current_total = sum(alloc.values())
    diff = top_n_total - current_total
    if diff > 0:
        candidates = [s for s in subs if alloc[s] < max_per_sub]
        idx = 0
        while diff > 0 and candidates:
            s = candidates[idx % len(candidates)]
            if alloc[s] < max_per_sub:
                alloc[s] += 1
                diff -= 1
            idx += 1
            candidates = [s for s in subs if alloc[s] < max_per_sub]
    elif diff < 0:
        shortage = -diff
        for s in sorted(subs, key=lambda s: raw_scores[s]):
            if shortage <= 0:
                break
            removable = alloc[s] - min_per_sub
            if removable <= 0:
                continue
            take = min(removable, shortage)
            alloc[s] -= take
            shortage -= take
    final_total = sum(alloc.values())
    if final_total != top_n_total:
        # final adjustment (best-effort)
        if final_total > top_n_total:
            excess = final_total - top_n_total
            for s in sorted(subs, key=lambda x: -alloc[x]):
                if excess <= 0:
                    break
                can_remove = alloc[s] - min_per_sub
                if can_remove <= 0:
                    continue
                remove = min(can_remove, excess)
                alloc[s] -= remove
                excess -= remove
        else:
            need = top_n_total - final_total
            for s in sorted(subs, key=lambda x: -raw_scores[x]):
                if need <= 0:
                    break
                can_add = max_per_sub - alloc[s]
                if can_add <= 0:
                    continue
                add = min(can_add, need)
                alloc[s] += add
                need -= add
    return alloc


# -------------------------
# Public: per-subreddit adaptive fetch (PRAW primary, simulator fallback)
# -------------------------
def fetch_reddit_user_posts_by_subreddit(
    user_id: str,
    top_n_total: int = 100,
    prefer_since_last_run: bool = True,
    min_per_sub: int = 5,
    max_per_sub: int = 40,
    sample_activity_limit: int = 12
) -> Dict[str, Any]:
    logger.info("[live_reddit] Starting per-subreddit fetch for user=%s top_n_total=%d", user_id, top_n_total)

    # get subreddits for user
    subs = []
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT subreddits FROM users WHERE user_id=%s", (user_id,))
                row = cur.fetchone()
                if row and row[0]:
                    subs = [s for s in row[0] if s]
    except Exception:
        logger.warning("[live_reddit] Could not read user subreddits from DB for user=%s", user_id)

    subs = list(dict.fromkeys([s.lower() for s in (subs or [])]))
    if not subs:
        subs = ["technology", "machinelearning", "datascience"]
        logger.info("[live_reddit] No subs found for user=%s — using defaults: %s", user_id, subs)

    # since logic (optional)
    since = None
    if prefer_since_last_run:
        try:
            with utils.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT last_run_at FROM live_runs WHERE user_id=%s AND source=%s", (user_id, "reddit"))
                    row = cur.fetchone()
                    if row and row[0]:
                        since = row[0]
                        if isinstance(since, str):
                            since = datetime.fromisoformat(since)
                        if since and since.tzinfo is None:
                            since = since.replace(tzinfo=timezone.utc)
        except Exception:
            logger.debug("[live_reddit] Could not determine last run")

    # estimate activity
    activity_map = {}
    for s in subs:
        try:
            activity_map[s] = _estimate_activity_for_subreddit(s, sample_limit=sample_activity_limit)
        except Exception:
            activity_map[s] = 1.0

    allocations = _allocate_posts_across_subreddits(activity_map, top_n_total, min_per_sub=min_per_sub, max_per_sub=max_per_sub)
    logger.info("[live_reddit] Allocations for user=%s : %s", user_id, allocations)

    results = {}
    all_posts = []
    for s, alloc in allocations.items():
        if alloc <= 0:
            results[s] = []
            continue
        fetched = []
        # prefer PRAW
        if praw and config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET and config.REDDIT_USER_AGENT:
            try:
                fetched = _fetch_via_praw_for_subreddit(s, limit=alloc, sort="top", since=since)
                if len(fetched) < alloc:
                    more = _fetch_via_praw_for_subreddit(s, limit=alloc - len(fetched), sort="new", since=since)
                    existing = {p["post_id"] for p in fetched}
                    for p in more:
                        if p["post_id"] not in existing:
                            fetched.append(p)
            except Exception as e:
                logger.warning("[live_reddit] PRAW fetch failed for %s: %s — using simulator", s, e)
                fetched = _simulate_posts_for_subreddit(s, limit=alloc, since=since)
        else:
            fetched = _simulate_posts_for_subreddit(s, limit=alloc, since=since)

        # filter by since
        if since:
            try:
                filtered = []
                for p in fetched:
                    try:
                        created = datetime.fromisoformat(p.get("created_at"))
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=timezone.utc)
                        if created > since:
                            filtered.append(p)
                    except Exception:
                        filtered.append(p)
                fetched = filtered
            except Exception:
                pass

        # cap
        if len(fetched) > alloc:
            fetched = fetched[:alloc]

        results[s] = fetched
        all_posts.extend(fetched)

    # persist
    try:
        _persist_posts_to_db(all_posts)
    except Exception as e:
        logger.exception("[live_reddit] Failed to persist posts: %s", e)

    meta = {
        "user_id": user_id,
        "requested_total": top_n_total,
        "actual_fetched_total": len(all_posts),
        "allocations": allocations,
        "activity_map": activity_map,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    logger.info("[live_reddit] Completed per-subreddit fetch for user=%s, fetched=%d", user_id, len(all_posts))
    return {"results": results, "meta": meta}


# -------------------------
# Backwards-compatible flat fetcher
# -------------------------
def fetch_reddit_user_posts(user_id: str, since: Optional[datetime] = None, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        out = fetch_reddit_user_posts_by_subreddit(user_id, top_n_total=limit, prefer_since_last_run=False)
        flat = []
        for posts in out.get("results", {}).values():
            flat.extend(posts)
        flat_sorted = sorted(flat, key=lambda p: p.get("created_at") or "", reverse=True)
        return flat_sorted[:limit]
    except Exception:
        # fallback simple simulator
        subs = ["technology", "machinelearning", "datascience"]
        out = []
        for s in subs:
            out.extend(_simulate_posts_for_subreddit(s, limit=int(limit / len(subs))))
        return out[:limit]


# convenience wrapper
def analyze_reddit_feed_per_subreddit(user_id: str, top_n_total: int = 100, prefer_since_last_run: bool = True):
    return fetch_reddit_user_posts_by_subreddit(user_id=user_id, top_n_total=top_n_total, prefer_since_last_run=prefer_since_last_run)


# quick debug
if __name__ == "__main__":
    import json
    print(json.dumps(fetch_reddit_user_posts_by_subreddit("user_1", top_n_total=40)["meta"], indent=2))
