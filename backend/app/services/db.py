import json
import sqlite3
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "backend" / "cropdoc.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnosis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                crop TEXT NOT NULL,
                disease TEXT NOT NULL,
                is_healthy INTEGER NOT NULL,
                confidence REAL NOT NULL,
                top3_json TEXT NOT NULL,
                advice_json TEXT NOT NULL,
                heatmap_url TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def save_diagnosis_record(
    filename: str,
    prediction: dict,
    top3: list,
    advice: dict,
    heatmap_url: str | None,
) -> int:
    created_at = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO diagnosis_records (
                filename,
                crop,
                disease,
                is_healthy,
                confidence,
                top3_json,
                advice_json,
                heatmap_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                prediction["crop"],
                prediction["disease"],
                1 if prediction["is_healthy"] else 0,
                prediction["confidence"],
                json.dumps(top3, ensure_ascii=False),
                json.dumps(advice, ensure_ascii=False),
                heatmap_url,
                created_at,
            ),
        )

        return cursor.lastrowid


def list_diagnosis_records(limit: int = 20) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                crop,
                disease,
                is_healthy,
                confidence,
                top3_json,
                advice_json,
                heatmap_url,
                created_at
            FROM diagnosis_records
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records = []

    for row in rows:
        records.append({
            "id": row["id"],
            "filename": row["filename"],
            "prediction": {
                "crop": row["crop"],
                "disease": row["disease"],
                "is_healthy": bool(row["is_healthy"]),
                "confidence": row["confidence"],
            },
            "top3": json.loads(row["top3_json"]),
            "advice": json.loads(row["advice_json"]),
            "explanation": {
                "gradcam_available": bool(row["heatmap_url"]),
                "heatmap_url": row["heatmap_url"],
            },
            "created_at": row["created_at"],
        })

    return records
