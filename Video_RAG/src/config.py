import os

BASE = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TRANSCRIPTION_MODEL = "whisper-1"
