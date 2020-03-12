print("Spast")

import tensorflow as tf
from bin import database

if __name__ == "__main__":
    db = database.db()
    db.loadaudio()