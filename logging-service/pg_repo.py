#!/usr/bin/env python3
"""
PostgreSQL connection utilities for AI agents repositories
"""

from __future__ import annotations

import os
import psycopg2
from typing import Optional


def _read_secret_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def get_db_password() -> str:
    # Prefer AI-specific file-based secret when available
    pw = _read_secret_file(os.getenv("AI_DB_PASSWORD_FILE"))
    if pw:
        return pw
    # Fallback to general DB password file
    pw = _read_secret_file(os.getenv("DB_PASSWORD_FILE"))
    if pw:
        return pw
    # Env var overrides
    return os.getenv("AI_DB_PASSWORD", os.getenv("DB_PASSWORD", "your_db_password"))


def get_pg_connection():
    """Create a psycopg2 connection using env vars.

    Env vars used (in order of precedence):
      - AI_DB_HOST, AI_DB_NAME, AI_DB_USER, AI_DB_PASSWORD_FILE/AI_DB_PASSWORD
      - DB_HOST, DB_NAME, DB_USER, DB_PASSWORD_FILE/DB_PASSWORD
    """
    host = os.getenv("AI_DB_HOST", os.getenv("DB_HOST", "postgres"))
    name = os.getenv("AI_DB_NAME", os.getenv("DB_NAME", "sap_data_quality"))
    user = os.getenv("AI_DB_USER", os.getenv("DB_USER", "sap_user"))
    password = get_db_password()

    conn = psycopg2.connect(
        host=host,
        dbname=name,
        user=user,
        password=password,
        port=5432,
    )
    # Autocommit simplifies simple CRUD operations
    conn.autocommit = True
    return conn


