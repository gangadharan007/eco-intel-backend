import os
import mysql.connector
from mysql.connector import Error
import logging

logger = logging.getLogger(__name__)

def get_connection():
    """
    ✅ PRODUCTION: Connects to Railway MySQL for Vercel deployment
    Uses env vars set in Vercel dashboard
    """
    try:
        conn = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "localhost"),
            port=int(os.environ.get("MYSQL_PORT", 3306)),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD"),
            database=os.environ.get("MYSQL_DATABASE", "railway"),
            autocommit=True,
        )
        logger.info("✅ DB Connected successfully")
        return conn
    except Error as e:
        logger.error(f"❌ DB Connection failed: {e}")
        return None

# Test connection function (for local debugging)
def test_connection():
    conn = get_connection()
    if conn:
        print("✅ Railway DB Connected!")
        conn.close()
    else:
        print("❌ DB Connection failed")

if __name__ == "__main__":
    test_connection()
