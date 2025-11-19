"""
stage4_global_analysis.py

Spark-first per-subreddit topic analysis + live persistence using
Mistral-7B-Instruct-v0.2 (4-bit via bitsandbytes if available).

Features:
 - ChatML-wrapped LLM calls (system/user/assistant) to prevent prompt-echo.
 - BitsAndBytes 4-bit load if available with safe fallbacks.
 - SBERT embeddings on driver (CUDA if available).
 - Spark KMeans preferred; sklearn KMeans fallback.
 - Per-subreddit adaptive allocation orchestration (uses live_reddit fetcher).
 - EDA hook integration with stage4_eda.
 - Persists outputs to live_* tables only.
"""

from __future__ import annotations

import json
import re
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import torch

from sentence_transformers import SentenceTransformer

# Spark
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F
from pyspark.sql import types as T

# sklearn fallback
from sklearn.cluster import KMeans as SkKMeans

# transformers + optional bitsandbytes
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

from src.core import config, utils
from src.core.logging_config import get_logger

logger = get_logger(__name__) or logging.getLogger(__name__)

_MODEL_CACHE: Dict[str, Any] = {}
SAMPLE_JOINER = " ||| "
DEFAULT_SBERT_BATCH = getattr(config, "SBERT_BATCH_SIZE", 32)
DEFAULT_FULL_PIPELINE_THRESHOLD = getattr(config, "FULL_PIPELINE_THRESHOLD", 20)

# Live-only schema (ensured by _ensure_live_tables)
LIVE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS live_outputs
(
    id           SERIAL PRIMARY KEY,
    user_id      TEXT,
    source       TEXT,
    created_at   TIMESTAMP DEFAULT now(),
    payload_json JSONB
);

CREATE TABLE IF NOT EXISTS live_runs
(
    user_id     TEXT,
    source      TEXT,
    last_run_at TIMESTAMP,
    post_count  INTEGER,
    output_id   INTEGER REFERENCES live_outputs (id),
    PRIMARY KEY (user_id, source)
);

CREATE TABLE IF NOT EXISTS live_topics
(
    id           SERIAL PRIMARY KEY,
    user_id      TEXT NOT NULL,
    source       TEXT NOT NULL,
    topic_title  TEXT,
    summary_text TEXT,
    sample_posts TEXT,
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS live_post_topic_mapping
(
    id         SERIAL PRIMARY KEY,
    user_id    TEXT NOT NULL,
    source     TEXT NOT NULL,
    post_ref   TEXT,
    topic_ref  INTEGER REFERENCES live_topics (id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

# -------------------------
# LLM loader & ChatML wrapper (Mistral 7B v0.2 default)
# -------------------------
def _get_llm_model_name() -> str:
    return getattr(config, "LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")


def get_quantized_llm_pipeline() -> Any:
    """
    Load a transformers pipeline configured for instruction-following using ChatML.
    Attempts 4-bit bnb (NF4) if available and CUDA is present. Falls back to fp16/gpu or cpu.
    Returns a text-generation pipeline object.
    """
    if "llm" in _MODEL_CACHE:
        return _MODEL_CACHE["llm"]

    model_name = _get_llm_model_name()
    use_cuda = torch.cuda.is_available()
    bnb_ok = _HAS_BNB and BitsAndBytesConfig is not None and use_cuda

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        logger.warning("[stage4] Tokenizer load failed (%s), trying without fast tokenizer.", e)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if bnb_ok:
        try:
            logger.info("[stage4] Attempting 4-bit bitsandbytes load (nf4) for %s", model_name)
            bnb_conf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_conf,
                device_map="auto",
                trust_remote_code=True
            )
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", return_full_text=False)
            _MODEL_CACHE["llm"] = pipe
            logger.info("[stage4] Loaded 4-bit LLM (bitsandbytes).")
            return pipe
        except Exception as e:
            logger.warning("[stage4] bitsandbytes load failed: %s — falling back.", e)

    # fallback: try fp16 on GPU or cpu
    try:
        if use_cuda:
            logger.info("[stage4] Loading LLM with fp16 on CUDA (fallback).")
            pipe = pipeline("text-generation", model=model_name, tokenizer=tokenizer, device=0, torch_dtype=torch.float16, return_full_text=False)
        else:
            logger.info("[stage4] Loading LLM on CPU (fallback).")
            pipe = pipeline("text-generation", model=model_name, tokenizer=tokenizer, return_full_text=False)
        _MODEL_CACHE["llm"] = pipe
        logger.info("[stage4] LLM pipeline ready (fallback).")
        return pipe
    except Exception as e:
        logger.exception("[stage4] Failed to initialize LLM pipeline: %s", e)
        raise RuntimeError("LLM pipeline initialization failed.") from e


def prompt_chatml(generator, system_msg: str, user_msg: str, max_new_tokens: int = 256) -> str:
    """
    Wrap system + user message into Mistral ChatML format and call pipeline.
    Returns a string (generated assistant text).
    """
    # ChatML template recommended for Mistral
    chat_input = "<|system|>\n" + system_msg.strip() + "\n<|user|>\n" + user_msg.strip() + "\n<|assistant|>\n"
    # Call pipeline with deterministic params
    out = generator(chat_input, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
    # pipeline returns list of dicts; pick generated_text or text
    if isinstance(out, list) and out:
        first = out[0]
        text = first.get("generated_text") or first.get("text") or ""
    elif isinstance(out, dict):
        text = out.get("generated_text") or out.get("text") or ""
    else:
        text = ""
    return text.strip()


def get_llm_pipeline() -> Any:
    return _MODEL_CACHE.get("llm") or get_quantized_llm_pipeline()


# -------------------------
# SBERT & Sentiment loaders (driver)
# -------------------------
def get_sentence_transformer():
    if "sbert" not in _MODEL_CACHE:
        logger.info("[stage4] Loading SentenceTransformer 'all-MiniLM-L6-v2' (driver).")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE["sbert"] = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info("[stage4] SBERT ready on %s.", device)
    return _MODEL_CACHE["sbert"]


def get_sentiment_pipeline():
    if "sentiment" not in _MODEL_CACHE:
        logger.info("[stage4] Loading sentiment-analysis pipeline (driver).")
        if torch.cuda.is_available():
            _MODEL_CACHE["sentiment"] = pipeline("sentiment-analysis", device=0)
        else:
            _MODEL_CACHE["sentiment"] = pipeline("sentiment-analysis")
        logger.info("[stage4] Sentiment pipeline ready.")
    return _MODEL_CACHE["sentiment"]


# -------------------------
# Text helpers & dedupe
# -------------------------
def _clean_text_for_model(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"http\S+", " ", text)
    t = re.sub(r"www\.\S+", " ", t)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = t.encode("ascii", "ignore").decode()
    return t.strip()


def restore_readable_text(text: str) -> str:
    return _clean_text_for_model(text or "")


def prepare_llm_text(texts: List[str], max_join_chars: int = 4000) -> str:
    cleaned = []
    for t in texts:
        if not t:
            continue
        t = t.strip()
        if not re.search(r"[.!?]$", t):
            t = t + "."
        t = t[0].upper() + t[1:] if len(t) > 0 else t
        cleaned.append(t)
    joined = " ".join(cleaned)
    return joined[:max_join_chars]


def _dedupe_sentences_by_embedding(sentences: List[str], sim_threshold: float = 0.84, top_k: Optional[int] = None) -> List[str]:
    if not sentences:
        return []
    # quick normalized exact dedupe
    normed = []
    seen = set()
    for s in sentences:
        s_norm = re.sub(r"\s+", " ", (s or "").strip().lower())
        if s_norm and s_norm not in seen:
            seen.add(s_norm)
            normed.append(s)
    sentences = normed
    try:
        sbert = get_sentence_transformer()
    except Exception:
        return sentences[:top_k] if top_k else sentences
    try:
        emb = sbert.encode(sentences, batch_size=DEFAULT_SBERT_BATCH, show_progress_bar=False, convert_to_numpy=True)
    except TypeError:
        emb = np.array(sbert.encode(sentences, batch_size=DEFAULT_SBERT_BATCH, show_progress_bar=False))
    keep = []
    used = np.zeros(len(sentences), dtype=bool)
    for i in range(len(sentences)):
        if used[i]:
            continue
        keep.append(sentences[i])
        vi = emb[i]
        if len(emb) == 1:
            used[i] = True
            continue
        sims = np.dot(emb, vi) / (np.linalg.norm(emb, axis=1) * (np.linalg.norm(vi) + 1e-12))
        similar_idx = np.where((sims >= sim_threshold) & (~used))[0]
        for j in similar_idx:
            if j == i:
                continue
            used[j] = True
        used[i] = True
        if top_k and len(keep) >= top_k:
            break
    return keep[:top_k] if top_k else keep


# -------------------------
# Summarization / title / bullets (ChatML + system/user prompts)
# -------------------------
def summarize_topic_text(generator, full_text: str, chunk_size: int = 2800) -> str:
    if not full_text:
        return "(No meaningful content available.)"
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    partial_summaries = []
    system_msg = (
        "You are an expert analyst. Summarize text concisely and factually. "
        "Do NOT mention posts, Reddit, or that this content comes from comments or titles. "
        "Avoid filler and repeated wording."
    )
    for chunk in chunks:
        user_msg = f"Summarize the following content into a single concise paragraph:\n\n{chunk}\n\nSummary:"
        try:
            part = prompt_chatml(generator, system_msg, user_msg, max_new_tokens=160)
            if part:
                part = re.sub(r"^\s*bullets:\s*", "", part, flags=re.IGNORECASE)
                partial_summaries.append(part.strip())
        except Exception as e:
            logger.warning("[stage4] chunk summarize failed: %s", e)
    if not partial_summaries:
        return "(Unable to generate summary.)"
    try:
        partial_summaries = _dedupe_sentences_by_embedding(partial_summaries, sim_threshold=0.86)
    except Exception:
        # keep unique
        uniq = []
        seen = set()
        for s in partial_summaries:
            n = re.sub(r"\s+", " ", s.strip().lower())
            if n not in seen:
                seen.add(n)
                uniq.append(s)
        partial_summaries = uniq
    joined = " ".join(partial_summaries)
    meta_system = "You are an expert editor. Combine and condense the partial summaries into one polished paragraph, removing repetition."
    meta_user = f"Partial summaries:\n\n{joined}\n\nFinal summary:"
    try:
        final = prompt_chatml(generator, meta_system, meta_user, max_new_tokens=220)
    except Exception as e:
        logger.warning("[stage4] meta summarization failed: %s", e)
        final = " ".join(partial_summaries)[:1200]
    # final sentence-level dedupe
    try:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', final) if s.strip()]
        sents = _dedupe_sentences_by_embedding(sents, sim_threshold=0.86)
        final = " ".join(sents).strip()
    except Exception:
        seen = set()
        out = []
        for s in re.split(r'(?<=[.!?])\s+', final):
            s = s.strip()
            if not s:
                continue
            nn = s.lower()
            if nn in seen:
                continue
            seen.add(nn)
            out.append(s)
        final = " ".join(out)
    return final[:1600] if final else "(No summary generated.)"


def extract_keywords(text: str, top_k: int = 8) -> List[str]:
    if not text:
        return []
    stopwords = {
        "the", "and", "a", "an", "to", "of", "for", "in", "on", "is", "are", "was", "were",
        "that", "this", "with", "it", "as", "by", "be", "from", "or", "at", "its", "has",
        "have", "will", "but", "not", "they", "their"
    }
    tokens = re.findall(r"\b[a-zA-Z0-9\-']+\b", text.lower())
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    if not tokens:
        return []
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(top_k)]


def generate_topic_title_from_summary(generator, summary_text: str, max_words: int = 7) -> str:
    keywords = extract_keywords(summary_text, top_k=8)
    kw_str = ", ".join(keywords) if keywords else summary_text[:200]
    system_msg = "Generate a short, descriptive 3-7 word title. Do not mention Reddit or posts. Be specific."
    user_msg = f"Summary:\n{summary_text}\n\nKeywords: {kw_str}\n\nTitle:"
    try:
        t = prompt_chatml(generator, system_msg, user_msg, max_new_tokens=16)
    except Exception as e:
        logger.warning("[stage4] title gen failed: %s", e)
        t = ""
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^A-Za-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    lower_t = t.lower()
    generic_patterns = [r"^list of", r"^this is", r"^a list of", r"^the list of", r"^bullets", r"^list:"]
    is_generic = any(re.search(p, lower_t) for p in generic_patterns) or len(t.split()) < 2
    if is_generic:
        if keywords:
            fallback = " ".join([k.title() for k in keywords[:max_words]])
            if fallback:
                return fallback
        first_sent = re.split(r'[.!?]', summary_text or "")[0].strip()
        if first_sent:
            words = [w for w in re.findall(r"\w+", first_sent) if len(w) > 2]
            if words:
                return " ".join(words[:max_words]).title()
        return "Topic Summary"
    return " ".join(t.split()[:max_words]).title()


def extract_bullets(generator, summary_text: str, num_bullets: int = 3) -> List[str]:
    system_msg = f"Extract exactly {num_bullets} clear, distinct bullet points (8-20 words each) from the summary. No meta-comments."
    user_msg = f"Summary:\n{summary_text}\n\nBullets:"
    raw_bullets: List[str] = []
    try:
        out = prompt_chatml(generator, system_msg, user_msg, max_new_tokens=160)
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'^[-•\*\s]+', '', line)
            if line.lower().startswith("bullets:"):
                line = line[len("bullets:"):].strip()
            line = re.sub(r"\s{2,}", " ", line).strip()
            if line:
                raw_bullets.append(line)
    except Exception as e:
        logger.warning("[stage4] bullet extraction failed: %s", e)
        raw_bullets = []
    if raw_bullets:
        try:
            bullets = _dedupe_sentences_by_embedding(raw_bullets, sim_threshold=0.84, top_k=num_bullets)
        except Exception:
            seen = set()
            bullets = []
            for b in raw_bullets:
                n = re.sub(r'\s+', ' ', b.strip().lower())
                if n not in seen:
                    seen.add(n)
                    bullets.append(b)
            bullets = bullets[:num_bullets]
    else:
        bullets = []
    if len(bullets) < num_bullets:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary_text) if s.strip()]
        candidate = []
        for s in sents:
            s2 = re.sub(r'^[^\w]+', '', s).strip()
            if len(s2.split()) < 3:
                continue
            candidate.append(s2)
        try:
            candidate = _dedupe_sentences_by_embedding(candidate, sim_threshold=0.84)
        except Exception:
            seen = set()
            uniq = []
            for c in candidate:
                n = c.lower()
                if n not in seen:
                    seen.add(n)
                    uniq.append(c)
            candidate = uniq
        for c in candidate:
            if len(bullets) >= num_bullets:
                break
            words = c.split()
            bullets.append(" ".join(words[:30]))
    final = [re.sub(r'\s{2,}', ' ', b).strip() for b in bullets][:num_bullets]
    return final


# -------------------------
# Evaluation metrics
# -------------------------
def evaluate_summary(original_texts: List[str], summary_text: str) -> Dict[str, Any]:
    orig_joined = " ".join(original_texts or [])
    orig_tokens = re.findall(r"\b[a-zA-Z0-9']+\b", orig_joined.lower())
    sum_tokens = re.findall(r"\b[a-zA-Z0-9']+\b", (summary_text or "").lower())
    orig_unique = set(orig_tokens)
    sum_unique = set(sum_tokens)
    orig_count = max(1, len(orig_tokens))
    sum_count = max(1, len(sum_tokens))
    compression_ratio = round(sum_count / orig_count, 4)
    overlap = round((len(sum_unique & orig_unique) / (len(orig_unique) or 1)), 4)
    metrics = {
        "compression_ratio": compression_ratio,
        "lexical_overlap": overlap,
        "summary_length": sum_count
    }
    return metrics


# -------------------------
# DB helpers & persistence (live-only)
# -------------------------
def _ensure_live_tables():
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(LIVE_SCHEMA_SQL)
                conn.commit()
        logger.debug("[stage4] Ensured live tables exist.")
    except Exception as e:
        logger.warning("[stage4] Could not ensure live tables: %s", e)


def _save_live_output(user_id: str, source: str, payload: Dict[str, Any]) -> int:
    payload_json = json.dumps(payload)
    with utils.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO live_outputs (user_id, source, payload_json) VALUES (%s, %s, %s) RETURNING id",
                        (user_id, source, payload_json))
            out_id = cur.fetchone()[0]
            conn.commit()
    return int(out_id)


def _upsert_live_run(user_id: str, source: str, post_count: int, output_id: int):
    now = datetime.now(timezone.utc)
    with utils.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO live_runs (user_id, source, last_run_at, post_count, output_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, source) DO UPDATE
                  SET last_run_at = EXCLUDED.last_run_at,
                      post_count  = EXCLUDED.post_count,
                      output_id   = EXCLUDED.output_id
                """, (user_id, source, now, post_count, output_id)
            )
            conn.commit()


def _persist_topics_and_mappings(user_id: str, source: str, topics: List[Dict[str, Any]], post_ids: List[str]) -> None:
    if not topics:
        return
    with utils.get_db_connection() as conn:
        with conn.cursor() as cur:
            topic_index_to_live_dbid = {}
            for t in topics:
                cur.execute(
                    "INSERT INTO live_topics (user_id, source, topic_title, summary_text, sample_posts) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (user_id or "", source or "", t.get("title"), t.get("summary"), SAMPLE_JOINER.join(t.get("sample_posts", []) or []))
                )
                live_dbid = cur.fetchone()[0]
                topic_index_to_live_dbid[int(t["topic_index"])] = live_dbid
            live_mapping_pairs = []
            for t in topics:
                tidx = int(t["topic_index"])
                l_dbid = topic_index_to_live_dbid.get(tidx)
                for post_i in t.get("post_indices", []):
                    pid = post_ids[int(post_i)]
                    live_mapping_pairs.append((pid, l_dbid, user_id or "", source or ""))
            if live_mapping_pairs:
                cur.executemany(
                    "INSERT INTO live_post_topic_mapping (post_ref, topic_ref, user_id, source) VALUES (%s, %s, %s, %s)",
                    live_mapping_pairs
                )
            conn.commit()


# -------------------------
# KMeans: Spark-first, sklearn fallback
# -------------------------
def _spark_kmeans_fit_predict(spark, np_emb: np.ndarray, n_clusters: int) -> np.ndarray:
    rows = [(int(i), Vectors.dense(vec.tolist())) for i, vec in enumerate(np_emb)]
    df = spark.createDataFrame(rows, schema=["index", "features"]).repartition(max(2, int(spark.sparkContext.defaultParallelism)))
    kmeans = SparkKMeans(k=n_clusters, seed=42, featuresCol="features", predictionCol="prediction", maxIter=20)
    model = kmeans.fit(df)
    preds = model.transform(df).select("index", "prediction").orderBy("index").collect()
    preds_arr = np.array([int(r["prediction"]) for r in preds], dtype=int)
    return preds_arr


def _sklearn_kmeans_predict(np_emb: np.ndarray, n_clusters: int) -> np.ndarray:
    km = SkKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    preds = km.fit_predict(np_emb)
    return preds


# -------------------------
# Heuristic for number of topics
# -------------------------
def _determine_num_topics_for_count(n_posts: int) -> int:
    if n_posts <= 10:
        return 1
    if n_posts <= 20:
        return 2
    if n_posts <= 40:
        return 3
    if n_posts <= 80:
        return 4
    return min(6, max(4, n_posts // 20))


# -------------------------
# Main per-subreddit orchestration (Spark-first)
# -------------------------
def run_subreddit_topic_analysis(
        user_id: str,
        top_n_total: int = 100,
        prefer_since_last_run: bool = True,
        full_pipeline_threshold: int = DEFAULT_FULL_PIPELINE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Orchestrate adaptive per-subreddit analysis for `user_id`.
    - Uses src.live.live_reddit.analyze_reddit_feed_per_subreddit to fetch grouped posts.
    - For each subreddit, runs topic modelling + summarization and persists results.
    """
    logger.info("[stage4] Starting subreddit-wise analysis for user=%s (top_n_total=%d)", user_id, top_n_total)
    _ensure_live_tables()

    try:
        from src.live import live_reddit
    except Exception:
        logger.error("[stage4] live_reddit module not available.")
        raise RuntimeError("live_reddit not available")

    fetched = live_reddit.analyze_reddit_feed_per_subreddit(user_id=user_id, top_n_total=top_n_total, prefer_since_last_run=prefer_since_last_run)
    results_by_sub = fetched.get("results", {})
    meta = fetched.get("meta", {})

    generator = get_llm_pipeline()
    sbert = get_sentence_transformer()
    sentiment_pipe = get_sentiment_pipeline()

    try:
        spark = utils.get_spark_session()
    except Exception:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local[*]").appName("stage4-subreddit-analysis").getOrCreate()

    overall_payload = {
        "user_id": user_id,
        "source": "reddit:subreddit_analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_n_total": top_n_total,
        "allocations": meta.get("allocations", {}),
        "subreddits": {}
    }

    enable_eda = getattr(config, "ENABLE_EDA", True)

    for subreddit, posts in results_by_sub.items():
        try:
            allocated = meta.get("allocations", {}).get(subreddit, 0)
            if not posts:
                overall_payload["subreddits"][subreddit] = {"allocated": allocated, "post_count": 0, "topics": []}
                continue

            texts = [restore_readable_text(p.get("content") or f"{p.get('title','')}\n\n{p.get('selftext','')}") for p in posts]
            post_ids = [p.get("post_id") or f"generated_{i}" for i, p in enumerate(posts)]
            n_posts = len(texts)
            logger.info("[stage4] Subreddit=%s posts=%d", subreddit, n_posts)

            num_topics = _determine_num_topics_for_count(n_posts)

            if n_posts <= full_pipeline_threshold:
                blob = prepare_llm_text(texts, max_join_chars=3500)
                summary = summarize_topic_text(generator, blob)
                title = generate_topic_title_from_summary(generator, summary)
                bullets = extract_bullets(generator, summary)
                sentiment = sentiment_pipe(summary[:512])[0] if summary else {"label": "NEUTRAL", "score": 0.0}
                metrics = evaluate_summary(texts, summary)
                topics = [{
                    "topic_index": 0,
                    "title": title,
                    "summary": summary,
                    "sample_posts": texts[:3],
                    "bullets": bullets,
                    "sentiment": sentiment,
                    "keywords": extract_keywords(summary, top_k=6),
                    "metrics": metrics,
                    "post_indices": list(range(n_posts))
                }]
                preds = np.zeros(n_posts, dtype=int)
            else:
                try:
                    embeddings = sbert.encode(texts, batch_size=DEFAULT_SBERT_BATCH, show_progress_bar=False, convert_to_numpy=True)
                except TypeError:
                    embeddings = np.array(sbert.encode(texts, batch_size=DEFAULT_SBERT_BATCH, show_progress_bar=False))
                np_emb = np.array(embeddings)

                try:
                    preds = _spark_kmeans_fit_predict(spark, np_emb, n_clusters=num_topics)
                except Exception:
                    logger.info("[stage4] Spark KMeans failed; using sklearn fallback.")
                    preds = _sklearn_kmeans_predict(np_emb, n_clusters=num_topics)

                topics = []
                for tid in range(int(num_topics)):
                    indices = [i for i, p in enumerate(preds) if int(p) == tid]
                    if not indices:
                        continue
                    posts_for_topic = [texts[i] for i in indices]
                    unique_posts = list(dict.fromkeys(posts_for_topic))
                    full_text = prepare_llm_text(unique_posts, max_join_chars=3500)
                    try:
                        summary = summarize_topic_text(generator, full_text)
                    except Exception as e:
                        logger.warning("[stage4] Summary generation failed for topic %d: %s", tid, e)
                        summary = "(Error generating summary.)"
                    title = generate_topic_title_from_summary(generator, summary)
                    bullets = extract_bullets(generator, summary)
                    sentiment = sentiment_pipe(summary[:512])[0] if summary else {"label": "NEUTRAL", "score": 0.0}
                    metrics = evaluate_summary(unique_posts, summary)
                    keywords = extract_keywords(summary, top_k=6)
                    topics.append({
                        "topic_index": int(tid),
                        "title": title,
                        "summary": summary,
                        "sample_posts": unique_posts[:3],
                        "bullets": bullets,
                        "sentiment": sentiment,
                        "keywords": keywords,
                        "metrics": metrics,
                        "post_indices": indices
                    })

            source_name = f"reddit:{subreddit}"
            payload = {
                "user_id": user_id,
                "source": source_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "post_count": n_posts,
                "result_topics": topics,
                "post_ids": post_ids
            }
            try:
                output_id = _save_live_output(user_id, source_name, payload)
                _upsert_live_run(user_id, source_name, n_posts, output_id)
            except Exception as e:
                logger.warning("[stage4] Could not save live output for %s: %s", subreddit, e)
                output_id = None

            try:
                _persist_topics_and_mappings(user_id, source_name, topics, post_ids)
            except Exception as e:
                logger.warning("[stage4] Could not persist topics/mappings for %s: %s", subreddit, e)

            # EDA per subreddit (simple spark aggregates)
            if enable_eda:
                try:
                    rows = []
                    for i, p in enumerate(posts):
                        rows.append((post_ids[i], restore_readable_text(p.get("content") or f"{p.get('title','')}\n\n{p.get('selftext','')}"),
                                     int(p.get("score") or 0), int(p.get("num_comments") or 0), str(p.get("created_at"))))
                    schema = T.StructType([
                        T.StructField("post_id", T.StringType(), True),
                        T.StructField("content", T.StringType(), True),
                        T.StructField("score", T.IntegerType(), True),
                        T.StructField("num_comments", T.IntegerType(), True),
                        T.StructField("created_at", T.StringType(), True)
                    ])
                    df_sub = spark.createDataFrame(rows, schema=schema)
                    sub_eda = {
                        "post_count": n_posts,
                        "avg_score": float(df_sub.agg(F.avg(F.col("score"))).first()[0] or 0.0),
                        "avg_comments": float(df_sub.agg(F.avg(F.col("num_comments"))).first()[0] or 0.0)
                    }
                except Exception as e:
                    logger.debug("[stage4] EDA per-subreddit failed for %s: %s", subreddit, e)
                    sub_eda = {}
            else:
                sub_eda = {}

            overall_payload["subreddits"][subreddit] = {
                "allocated": allocated,
                "post_count": n_posts,
                "output_id": output_id,
                "topics": [{
                    "topic_index": t["topic_index"],
                    "title": t["title"],
                    "bullets": t.get("bullets", []),
                    "sample_posts": t.get("sample_posts", []),
                    "metrics": t.get("metrics", {})
                } for t in topics],
                "eda": sub_eda
            }

        except Exception as e:
            logger.exception("[stage4] Error processing subreddit %s: %s", subreddit, e)
            overall_payload["subreddits"][subreddit] = {"error": str(e)}

    # Save combined run record
    try:
        combined_output_id = _save_live_output(user_id, "reddit:subreddit_analysis", overall_payload)
        _upsert_live_run(user_id, "reddit:subreddit_analysis",
                         sum([v.get("post_count", 0) if isinstance(v, dict) else 0 for v in overall_payload["subreddits"].values()]),
                         combined_output_id)
    except Exception as e:
        logger.debug("[stage4] Could not save combined subreddit analysis output: %s", e)

    logger.info("[stage4] Completed subreddit-wise analysis for user=%s", user_id)
    return {"status": "ok", "payload": overall_payload}
