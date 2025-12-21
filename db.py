import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Gangadharan@kgcas",
        database="eco_intel_ai"
    )
