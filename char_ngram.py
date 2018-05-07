import numpy
import os
import torch
import torch.nn as nn
import pickle
from torch.autograd import Variable
import argparse
from util import *
from model import *
from tqdm import tqdm

class ParaphraseData:
    def __init__(self, raw_text, label):
        self.raw_text = raw_text
        self.label = label

    def set_gram_counter(self):
        gram_list = list(itertools.chain.from_iterable(self.ngrams_a))
        gram_list += list(itertools.chain.from_iterable(self.ngrams_b))
        self.gram_counter = Counter(gram_list)

    def set_ngram(self, ngrams_a, ngrams_b):
        self.ngrams_a = ngrams_a
        self.ngrams_b = ngrams_b

    def set_ngrams_idx(self, ngrams_idx_a, ngrams_idx_b):
        self.ngrams_idx_a = tailor(ngrams_idx_a)
        self.ngrams_idx_b = tailor(ngrams_idx_b)


def tailor(ngrams_idx):
    padded_idx = []

    sentence_length = len(ngrams_idx)
    for idxs in ngrams_idx:
        gram_length = len(idxs)
        if gram_length >= MAX_PHRASES_LENGTH:
            idxs = idxs[0: MAX_PHRASES_LENGTH]
            padded_idx.append(idxs)
        else:
            idxs = list(np.pad(idxs, (0, MAX_PHRASES_LENGTH - gram_length), 'constant'))
            padded_idx.append(idxs)
    if len(padded_idx) < MAX_SENTENCE_LENGTH:
        padded_idx += [[0] * MAX_PHRASES_LENGTH for i in range(MAX_SENTENCE_LENGTH - sentence_length)]
    else:
        padded_idx = padded_idx[0: MAX_SENTENCE_LENGTH]
    return padded_idx

def construct_data_set(file_name):
    print("constructing data set for some some{0}".format(file_name))
    train_data_set = []
    test_data_set = []
    dev_data_set = []
    paraphrase_df = pd.read_csv(file_name, header=None, names=["id", "qid1", "qid2", "text_a", "text_b", "label"], na_filter=False)
    train_paraphrase_df = paraphrase_df.iloc[0:50000,:] 
    test_paraphrase_df  = paraphrase_df.iloc[50000:60000,:] 
    dev_paraphrase_df   = paraphrase_df.iloc[60000:70000,:]

    for index in tqdm(range(len(train_paraphrase_df))):
        label = train_paraphrase_df.iloc[index, :]['label']
        text_a = train_paraphrase_df.iloc[index, :]['text_a']
        text_b = train_paraphrase_df.iloc[index, :]['text_b']
        text = (text_a, text_b)
        paraphrase_data = ParaphraseData(raw_text=text, label=label)
        train_data_set.append(paraphrase_data)

    for index in tqdm(range(len(test_paraphrase_df))):
        label = test_paraphrase_df.iloc[index, :]['label']
        text_a = test_paraphrase_df.iloc[index, :]['text_a']
        text_b = test_paraphrase_df.iloc[index, :]['text_b']
        text = (text_a, text_b)
        paraphrase_data = ParaphraseData(raw_text=text, label=label)
        test_data_set.append(paraphrase_data)

    for index in tqdm(range(len(dev_paraphrase_df))):
        label = dev_paraphrase_df.iloc[index, :]['label']
        text_a = dev_paraphrase_df.iloc[index, :]['text_a']
        text_b = dev_paraphrase_df.iloc[index, :]['text_b']
        text = (text_a, text_b)
        paraphrase_data = ParaphraseData(raw_text=text, label=label)
        dev_data_set.append(paraphrase_data)

    return (train_data_set, test_data_set, dev_data_set)


def preprocess_text(text):
    text = text.translate(translator).lower()
    return text

def extract_ngram_from_phrases(phrases, n):
    phrases_ngram_list = []
    for phrase in phrases:
        word_ngram_list = []
        for word in phrase:
            padded_word = "[" + word + "]"
            if len(padded_word) < n:
                word_ngram_list.append([tuple(list(padded_word))])
            char_ngrams = list(zip(*[padded_word[i:] for i in range(n)]))
            if char_ngrams:
                word_ngram_list.append(char_ngrams)
        phrases_ngram_list.append(list(itertools.chain.from_iterable(word_ngram_list)))
    return phrases_ngram_list

def get_overlap_phrases(window_size, text):
    phrases_list = []
    word_list = text.split()
    sentence_length = len(word_list)
    for word_offset in range(sentence_length):
        index = [i for i in range(word_offset - window_size, word_offset + window_size + 1) if
                 i >= 0 and i < sentence_length]
        phrases_list.append([word_list[ind] for ind in index])
    return phrases_list

def extract_phrases_from_text(text, window_size):
    text_a, text_b = text[0], text[1]
    phrases_a = get_overlap_phrases(window_size, text_a)
    phrases_b = get_overlap_phrases(window_size, text_b)
    return phrases_a, phrases_b


def extract_ngram_from_text(text, window_size, n):
    phrases_a, phrases_b = extract_phrases_from_text(text, window_size)
    phrases_ngrams_a = extract_ngram_from_phrases(phrases_a, n)
    phrases_ngrams_b = extract_ngram_from_phrases(phrases_b, n)
    return phrases_ngrams_a, phrases_ngrams_b

def process_text_dataset(dataset, window_size, n, topk=None, ngram_indexer=None):
    for i in tqdm(range(len(dataset))):
        try:
            text = dataset[i].raw_text
            ngrams_a, ngrams_b = extract_ngram_from_text(text, window_size, n)
            dataset[i].set_ngram(ngrams_a=ngrams_a, ngrams_b=ngrams_b)
            if not ngram_indexer:
                dataset[i].set_gram_counter()
        except:
            print(i)

    if ngram_indexer is None:
        ngram_indexer = construct_ngram_indexer([datum.gram_counter for datum in dataset], topk)

    for i in tqdm(range(len(dataset))):
        dataset[i].set_ngrams_idx(ngrams_idx_a=phrases_ngrams_to_index(dataset[i].ngrams_a, ngram_indexer),
                                  ngrams_idx_b=phrases_ngrams_to_index(dataset[i].ngrams_b, ngram_indexer))

    return dataset, ngram_indexer


def construct_ngram_indexer(ngram_counter_list, topk):
    ngram_counter = Counter()
    for counter in tqdm(ngram_counter_list):
        ngram_counter.update(counter)
    ngram_counter_topk = ngram_counter.most_common(topk)
    ngram_indexer = {ngram_counter_topk[index][0]: index + 1 for index in range(len(ngram_counter_topk))}
    return ngram_indexer

def phrases_ngrams_to_index(phrases_ngrams_list, ngram_indexer):
    index_list = [[ngram_indexer[token] for token in phrases_ngrams if token in ngram_indexer]
                  for phrases_ngrams in phrases_ngrams_list]
    return index_list

train_data_dir = "../data/Quora_question_pair_partition/"
window_size = 1
ngram_n = 3
vocab_size = 20000
train_data_set, test_data_set, dev_data_set = construct_data_set(train_data_dir)
processed_train_data, gram_indexer = process_text_dataset(train_data_set, window_size,
                                                                  ngram_n, vocab_size)
processed_dev, _ = process_text_dataset(dev_data_set, window_size, ngram_n,
                                            ngram_indexer=gram_indexer)
processed_test, _ = process_text_dataset(test_data_set, window_size, ngram_n,
                                                 ngram_indexer=gram_indexer)