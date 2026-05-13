import sqlite3
import json
from .fusion import Seed

class SeedStorage:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        # Revised Schema: Auto-incrementing ID, separate unique identifier column
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT UNIQUE,
                content TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def save_seed(self, identifier, content, metadata):
        try:
            # Insert identifier, content, metadata. ID is auto-generated (1, 2, 3...)
            self.conn.execute(
                "INSERT INTO seeds (identifier, content, metadata) VALUES (?, ?, ?)", 
                (identifier, content, json.dumps(metadata))
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Identifier must be unique
            print(f"DB Info: Seed with identifier '{identifier}' already exists.")
        except Exception as e:
            print(f"DB Error: {e}")

    def get_existing_identifiers(self):
        """Returns a set of all identifiers currently in the database."""
        try:
            cursor = self.conn.execute("SELECT identifier FROM seeds")
            return {row[0] for row in cursor}
        except Exception as e:
            print(f"DB Read Error: {e}")
            return set()

    def get_seeds_as_objects(self):
        try:
            # Fetch numeric ID as well
            cursor = self.conn.execute("SELECT id, content, metadata FROM seeds")
            results = []
            for row in cursor:
                # Convert numeric DB ID to string for Seed object compatibility
                seed_id = str(row[0])
                content = row[1]
                metadata = json.loads(row[2]) if row[2] else {}
                
                results.append(Seed(id=seed_id, content=content, metadata=metadata))
            return results
        except Exception as e:
            print(f"DB Read Error: {e}")
            return []