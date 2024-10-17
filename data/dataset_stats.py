import os
import numpy as np
from datasets import Dataset, concatenate_datasets
import pandas as pd


def extract_stats():
    av_mood = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'average_mood.csv'))
    base_dir = "/share/cp/temp/musicmed/metadata_test_32k/"
    arrow_files = [os.path.join(base_dir, fn) for fn in os.listdir(base_dir) if
                   fn.endswith(".arrow") and fn.startswith("data")]
    ds = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])
    i = np.unique(np.concatenate(np.char.split(ds["INSTR"], ",")))
    i = i[i != ""]
    g = np.unique(np.concatenate(np.char.split(ds["GENRE"], ",")))
    g = g[g != ""]
    i_oh = np.zeros((10459, 41), dtype=bool)
    g_oh = np.zeros((10459, 95), dtype=bool)
    emotion_oh = np.zeros((10459, 14), dtype=bool)
    un_emotion = av_mood["emotion"].unique()
    inst = np.char.split(ds["INSTR"], ",")
    genre = np.char.split(ds["GENRE"], ",")
    moods = np.char.split(ds["TAGS"], ",")
    for index in range(len(ds)):
        instruments = inst[index]
        gg = genre[index]
        m = moods[index]
        for mood in m:
            if mood != "":
                em = av_mood[av_mood["mood"] == mood]["emotion"].to_list()[0]
                emotion_oh[index][em == un_emotion] = 1
        for instr in instruments:
            i_oh[index][instr == i] = 1
        for gen in gg:
            g_oh[index][gen == g] = 1

    emotion_df = pd.DataFrame(data=emotion_oh, columns=un_emotion)
    genre_df = pd.DataFrame(data=g_oh, columns=g)
    instrument_df = pd.DataFrame(data=i_oh, columns=i)
    df = pd.concat((emotion_df, genre_df, instrument_df), keys=["emotion", "genre", "instrument"], axis=1)

    df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'jamendo_stats.csv'))


def read_stats():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'jamendo_stats.csv'), header=[0,1])



if __name__ == "__main__":
    read_stats()

