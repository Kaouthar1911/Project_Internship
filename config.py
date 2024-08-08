import os

class Config:
    DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    SECRET_KEY = os.environ.get('SECRET_KEY')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
    PERSIST_DIR = "./st"
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3-70b-8192"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 20
    DATA_DIR = "./data"