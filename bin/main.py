import tensorflow as tf
import database

if __name__ == "__main__":
    db = database.db()
    index_arr = [0,1,2]
    db.loadaudio(index_arr)
