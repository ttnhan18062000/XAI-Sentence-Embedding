import torch
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from sentence_transformers import util

w_tokenizer = word_tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Function to encode sentence pairs
def encode(model, sent_pairs):
    all_sents = [
        sent for pair in sent_pairs for sent in pair
    ]  # Flatten list of sentence pairs
    embds = model.encode(all_sents, show_progress_bar=False)

    scores = []
    s1_embeds = []
    s2_embeds = []
    for i in range(len(sent_pairs)):
        s1_embeds.append(embds[2 * i])
        s2_embeds.append(embds[2 * i + 1])
        scores.append(float(util.pytorch_cos_sim(embds[2 * i], embds[2 * i + 1])[0][0]))

    return np.array(scores), s1_embeds, s2_embeds


# Tokenize sentences
def _tokenize_sent(sentence):
    if isinstance(sentence, str):
        tokens = w_tokenizer(sentence)
    elif isinstance(sentence, list):
        tokens = sentence

    return tokens


# Build features for given sentence pair
def build_feature(sent1, sent2):
    tokens1 = _tokenize_sent(sent1)
    tokens2 = _tokenize_sent(sent2)
    s1len = len(tokens1)
    s2len = len(tokens2)

    tdict = {"s1_{}".format(i): token for i, token in enumerate(tokens1)}
    tdict.update({"s2_{}".format(i): token for i, token in enumerate(tokens2)})

    return pd.DataFrame(tdict, index=[0]), s1len, s2len


def tokenize(s):
    return tokenizer.tokenize(s)
