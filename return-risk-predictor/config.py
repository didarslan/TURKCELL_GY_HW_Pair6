import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'nortwind'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
} 