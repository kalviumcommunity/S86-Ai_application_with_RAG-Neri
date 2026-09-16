import json
import sqlite3
from datetime import datetime

from src.config import BASE_DIR


DATABASE_PATH = BASE_DIR / "storage" / "neri_history.db"


def _get_connection():
    """Create a connection to the Neri history database."""

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """Create and update the Neri database tables."""

    connection = _get_connection()

    # --------------------------------------------------
    # Troubleshooting History Table
    # --------------------------------------------------

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS troubleshooting_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            problem TEXT NOT NULL,
            error_code TEXT,
            response TEXT NOT NULL,
            feedback TEXT,
            feedback_comment TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    # Check whether this is an older database.
    columns = connection.execute(
        "PRAGMA table_info(troubleshooting_history)"
    ).fetchall()

    column_names = {
        column["name"]
        for column in columns
    }

    # Add feedback column if it does not exist.
    if "feedback" not in column_names:
        connection.execute(
            """
            ALTER TABLE troubleshooting_history
            ADD COLUMN feedback TEXT
            """
        )

    # Add feedback comment column if it does not exist.
    if "feedback_comment" not in column_names:
        connection.execute(
            """
            ALTER TABLE troubleshooting_history
            ADD COLUMN feedback_comment TEXT
            """
        )

    # --------------------------------------------------
    # Documents Table
    # --------------------------------------------------

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            document_type TEXT NOT NULL,
            machine TEXT,
            version TEXT,
            owner TEXT,
            chunks INTEGER NOT NULL,
            status TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )
        """
    )

    # Check whether the documents table is an older version.
    document_columns = connection.execute(
        "PRAGMA table_info(documents)"
    ).fetchall()

    document_column_names = {
        column["name"]
        for column in document_columns
    }

    # Add machine column if it does not exist.
    if "machine" not in document_column_names:
        connection.execute(
            """
            ALTER TABLE documents
            ADD COLUMN machine TEXT
            """
        )

    # Add version column if it does not exist.
    if "version" not in document_column_names:
        connection.execute(
            """
            ALTER TABLE documents
            ADD COLUMN version TEXT
            """
        )

    # Add owner column if it does not exist.
    if "owner" not in document_column_names:
        connection.execute(
            """
            ALTER TABLE documents
            ADD COLUMN owner TEXT
            """
        )

    connection.commit()
    connection.close()


# ======================================================
# Troubleshooting History
# ======================================================


def save_session(
    machine: str,
    machine_id: str,
    problem: str,
    error_code: str | None,
    response: dict,
) -> int:
    """Save a troubleshooting session."""

    initialize_database()

    connection = _get_connection()

    cursor = connection.execute(
        """
        INSERT INTO troubleshooting_history (
            machine,
            machine_id,
            problem,
            error_code,
            response,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            machine,
            machine_id,
            problem,
            error_code,
            json.dumps(response),
            datetime.now().isoformat(),
        ),
    )

    connection.commit()

    session_id = cursor.lastrowid

    connection.close()

    return session_id


def get_history(
    limit: int = 50,
) -> list[dict]:
    """Return recent troubleshooting sessions."""

    initialize_database()

    connection = _get_connection()

    rows = connection.execute(
        """
        SELECT
            id,
            machine,
            machine_id,
            problem,
            error_code,
            feedback,
            feedback_comment,
            created_at
        FROM troubleshooting_history
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_session(
    session_id: int,
) -> dict | None:
    """Return one complete troubleshooting session."""

    initialize_database()

    connection = _get_connection()

    row = connection.execute(
        """
        SELECT *
        FROM troubleshooting_history
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()

    connection.close()

    if row is None:
        return None

    session = dict(row)

    session["response"] = json.loads(
        session["response"]
    )

    return session


def save_feedback(
    session_id: int,
    feedback: str,
    feedback_comment: str | None = None,
) -> bool:
    """Save feedback for a troubleshooting session."""

    if feedback not in {
        "helpful",
        "not_helpful",
    }:
        raise ValueError(
            "Feedback must be 'helpful' or 'not_helpful'."
        )

    initialize_database()

    connection = _get_connection()

    cursor = connection.execute(
        """
        UPDATE troubleshooting_history
        SET
            feedback = ?,
            feedback_comment = ?
        WHERE id = ?
        """,
        (
            feedback,
            feedback_comment,
            session_id,
        ),
    )

    connection.commit()

    updated = cursor.rowcount > 0

    connection.close()

    return updated


# ======================================================
# Document Management
# ======================================================


def save_document(
    filename: str,
    document_type: str,
    chunks: int,
    machine: str | None = None,
    version: str | None = None,
    owner: str | None = None,
    status: str = "indexed",
) -> int:
    """Save uploaded document metadata."""

    initialize_database()

    connection = _get_connection()

    cursor = connection.execute(
        """
        INSERT OR REPLACE INTO documents (
            filename,
            document_type,
            machine,
            version,
            owner,
            chunks,
            status,
            uploaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            document_type,
            machine,
            version,
            owner,
            chunks,
            status,
            datetime.now().isoformat(),
        ),
    )

    connection.commit()

    document_id = cursor.lastrowid

    connection.close()

    return document_id


def get_documents(
    limit: int = 100,
) -> list[dict]:
    """Return uploaded document metadata."""

    initialize_database()

    connection = _get_connection()

    rows = connection.execute(
        """
        SELECT
            id,
            filename,
            document_type,
            machine,
            version,
            owner,
            chunks,
            status,
            uploaded_at
        FROM documents
        ORDER BY uploaded_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]