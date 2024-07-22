import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
from tqdm import tqdm

def sentence_to_words(sentence):
    import re
    
    tokens = re.findall(r"\b\w+\b|[.,!?;]", sentence)
    return tokens

def yahoo_csv_to_dict(data_path):

    df = pd.read_csv(data_path, header=None)

    inputs = df.iloc[:, 1:].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).values
    labels = df.iloc[:, 0].values

    data = []

    for i in range(len(inputs)):

        words = sentence_to_words(inputs[i])
        data.append({'label': labels[i] - 1, 'sent_a': words})

    return data
