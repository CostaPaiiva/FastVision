import os
import sqlite3
from typing import List, Tuple

DB_PATH = os.path.join("data", "app.db")

def init_db() -> None:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            FOREIGN KEY(person_id) REFERENCES people(id)
        )
    """)
    conn.commit()
    conn.close()

def upsert_person(name: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO people(name) VALUES(?)", (name,))
    conn.commit()
    cur.execute("SELECT id FROM people WHERE name=?", (name,))
    pid = cur.fetchone()[0]
    conn.close()
    return pid

def add_image(person_id: int, path: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO images(person_id, path) VALUES(?,?)", (person_id, path))
    conn.commit()
    conn.close()

def list_people() -> List[Tuple[int, str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM people ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return rows

def list_images() -> List[Tuple[int, int, str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, person_id, path FROM images")
    rows = cur.fetchall()
    conn.close()
    return rows

def delete_person(person_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM images WHERE person_id=?", (person_id,))
    cur.execute("DELETE FROM people WHERE id=?", (person_id,))
    conn.commit()
    conn.close()
