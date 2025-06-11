import sqlite3
import os

def ingest_sql_dump(sql_filepath: str, db_filepath: str):
    """
    Ingests SQL commands from a .sql dump file into a local SQLite3 database.

    This function reads SQL statements (CREATE TABLE, INSERT, etc.) from the
    specified .sql file and executes them sequentially against the SQLite3
    database. If the database file does not exist, it will be created.

    Args:
        sql_filepath (str): The path to the .sql dump file.
        db_filepath (str): The path to the SQLite3 database file (e.g., 'my_database.db').
                           If the file does not exist, it will be created.

    Returns:
        bool: True if the ingestion was successful, False otherwise.
    """
    if not os.path.exists(sql_filepath):
        print(f"Error: SQL dump file not found at '{sql_filepath}'")
        return False

    try:
        # Connect to the SQLite database. If it doesn't exist, it will be created.
        # isolation_level=None means autocommit mode, which is generally fine
        # for dump files as each statement is often self-contained.
        # However, for a multi-statement script like a dump, it's safer to
        # manage transactions manually or use executescript which handles it.
        conn = sqlite3.connect(db_filepath)
        cursor = conn.cursor()

        # Read the entire SQL dump file
        with open(sql_filepath, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        # Execute all SQL commands in the script.
        # executescript() is designed for executing multiple SQL statements
        # separated by semicolons, making it ideal for dump files.
        cursor.executescript(sql_script)

        # Commit the changes (executescript might autocommit or commit on success,
        # but an explicit commit here ensures everything is saved).
        conn.commit()

        print(f"Successfully ingested SQL dump from '{sql_filepath}' into '{db_filepath}'")
        return True

    except sqlite3.Error as e:
        print(f"SQLite error during ingestion: {e}")
        # Rollback in case of an error to prevent partial writes
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        # Ensure the database connection is closed
        if conn:
            conn.close()