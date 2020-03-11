import numpy as np
import torch
import torchaudio
import pandas as pd
import re


class Database:
    def __init__(self):
        self.path = "../data/train.tsv"
        self.df = pd.read_csv(self.path, sep="\t")
        self.sentences = self.df["sentence"]
        self.paths = self.df["path"]

    def preprocess_sentences(self):
        self.df["sentence"] = self.df["sentence"].apply(lambda x: x.lower())
        self.df["sentence"] = self.df["sentence"].apply(lambda x: re.sub("[\\\"\'!?.\/]", "", x))
        self.df["sentence"] = self.df["sentence"].apply(lambda x: x.split())
        self.sentences = self.df["sentence"]
        return None

    def load_sound(self, index):
        test_audio = torchaudio.load(audio_path + self.paths[index])
        return test_audio


if __name__ == "__main__":
    audio_path = "../data/clips/"
    db = Database()
    print(db.load_sound(2)[0].size())
    print(db.load_sound(2))
