import os
import logging
from urllib.parse import urlparse

import mysql.connector
from mysql.connector import Error

logger = logging.getLogger(__name__)


def _connect_from_url(mysql_url: str):
    """
    Parse: mysql://user:pass@host:port/dbname
    into mysql.connector.connect(host, port, user, password, database).
    """
    u = urlparse(mysql_url)
    db_name = (u.path or "").lstrip("/")

    if not u.hostname or not u.username or not db_name:
        raise ValueError("Invalid MYSQL_URL. Expected mysql://user:pass@host:port/dbname")

    return mysql.connector.connect(
        host=u.hostname,
        port=u.port or 3306,
        user=u.username,
        password=u.password,
        database=db_name,
        autocommit=True,  # supported connection argument [web:588]
    )


def get_connection():
    """
    Connect to Railway MySQL.

    Supports:
    - MYSQL_URL=mysql://user:pass@host:port/db
    - or MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    """
    try:
        mysql_url = os.environ.get("MYSQL_URL")
        if mysql_url:
            conn = _connect_from_url(mysql_url)
            logger.info("✅ DB Connected (MYSQL_URL)")
            return conn

        conn = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "localhost"),
            port=int(os.environ.get("MYSQL_PORT", 3306)),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD"),
            database=os.environ.get("MYSQL_DATABASE", "railway"),
            autocommit=True,
        )
        logger.info("✅ DB Connected (MYSQL_HOST vars)")
        return conn

    except Error as e:
        logger.error(f"❌ DB Connection failed (mysql error): {e}")
        return None
    except Exception as e:
        logger.error(f"❌ DB Connection failed: {e}")
        return None


def test_connection():
    conn = get_connection()
    if not conn:
        print("❌ DB Connection failed")
        return

    try:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        print("✅ Railway DB Connected! SELECT 1 =>", cur.fetchone())
        cur.close()
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_connection()
