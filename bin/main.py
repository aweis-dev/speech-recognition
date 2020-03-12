import tensorflow as tf
import database

if __name__ == "__main__":
    db = database.db()
    db.loadaudio()
