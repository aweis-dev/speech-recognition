import tensorflow as tf
import database

if __name__ == "__main__":
    db = database.db()
    index_arr = [0,1,2]
    audio_tensor = db.loadAudio(index_arr)
    true_values = db.getTrueValues(index_arr)
    print(audio_tensor)
    print(true_values)
