# wikirag/database.py
"""SQLite operations for raw article storage."""

import sqlite3
import os
from datetime import datetime
from lib.config import DB_PATH

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE NOT NULL,
                entity_type TEXT NOT NULL CHECK(entity_type IN ('person', 'place')),
                content TEXT NOT NULL,
                url TEXT,
                fetched_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def article_exists(title: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM articles WHERE title = ?", (title,)
        ).fetchone()
    return row is not None


def get_article(title: str):
    """Fetch a single article row by title. Returns sqlite3.Row or None."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT title, entity_type, content, url FROM articles WHERE title = ?",
            (title,)
        ).fetchone()


def save_article(title: str, entity_type: str, content: str, url: str):
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO articles (title, entity_type, content, url, fetched_at)
            VALUES (?, ?, ?, ?, ?)
        """, (title, entity_type, content, url, datetime.utcnow().isoformat()))
        conn.commit()


def get_all_articles():
    with get_connection() as conn:
        return conn.execute(
            "SELECT title, entity_type, content, url FROM articles"
        ).fetchall()


def get_articles_by_type(entity_type: str):
    with get_connection() as conn:
        return conn.execute(
            "SELECT title, entity_type, content, url FROM articles WHERE entity_type = ?",
            (entity_type,)
        ).fetchall()


def log_ingestion(title: str, status: str, message: str = ""):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO ingestion_log (title, status, message, created_at)
            VALUES (?, ?, ?, ?)
        """, (title, status, message, datetime.utcnow().isoformat()))
        conn.commit()
