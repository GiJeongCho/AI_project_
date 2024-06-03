import os

TEXT_MIN_LENGTH = int(os.getenv("TEXT_MIN_LENGTH", 1))
TEXT_MAX_LENGTH = int(os.getenv("TEXT_MAX_LENGTH", 10000))
