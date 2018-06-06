# coding:utf-8

# Quora data preparation
from __future__ import print_function

import numpy as np
import csv, json
from zipfile import ZipFile
from os.path import expanduser, exists

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file

# KERAS_DATASETS_DIR = expanduser('~/.keras/datasets/')
# QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
QUESTION_PAIRS_FILE = 'quora_duplicate_questions.tsv'
# GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
# GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
# GLOVE_FILE = 'glove.840B.300d.txt'

DATA_DIR = str( "../data/")
Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'


MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300



# if not exists(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE):
#     get_file(QUESTION_PAIRS_FILE, QUESTION_PAIRS_FILE_URL)
# print("Processing", QUESTION_PAIRS_FILE)

question1 = []
question2 = []
is_duplicate = []
# with open(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE, encoding='utf-8') as csvfile:
with open( "../data/quora_data/quora_duplicate_questions.tsv", encoding="utf-8"  ) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
        question1.append(row['question1'])
        question2.append(row['question2'])
        is_duplicate.append(row['is_duplicate'])
print('Question pairs: %d' % len(question1))




questions = question1 + question2
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)
question1_word_sequences = tokenizer.texts_to_sequences(question1)
question2_word_sequences = tokenizer.texts_to_sequences(question2)
word_index = tokenizer.word_index
print("Words in index: %d" % len(word_index))


# if not exists(KERAS_DATASETS_DIR + GLOVE_ZIP_FILE):
#     zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))
#     zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)
# print("Processing", GLOVE_FILE)
embeddings_index = {}
# with open(KERAS_DATASETS_DIR + GLOVE_FILE, encoding='utf-8') as f:
with open( "../data/vocab/glove.840B.300d.txt", encoding='utf-8' ) as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding
print('Word embeddings: %d' % len(embeddings_index))




nb_words = min(MAX_NB_WORDS, len(word_index))
word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))


q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(is_duplicate, dtype=int)
print('Shape of question1 data tensor:', q1_data.shape)
print('Shape of question2 data tensor:', q2_data.shape)
print('Shape of label tensor:', labels.shape)



np.save(open(DATA_DIR + Q1_TRAINING_DATA_FILE, 'wb'), q1_data)
np.save(open(DATA_DIR + Q2_TRAINING_DATA_FILE, 'wb'), q2_data)
np.save(open(DATA_DIR + LABEL_TRAINING_DATA_FILE, 'wb'), labels)
np.save(open(DATA_DIR + WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)

with open( DATA_DIR + NB_WORDS_DATA_FILE, 'w') as f:
    json.dump({'nb_words': nb_words}, f)

