import csv
from pickle import STOP
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# vocab_size = 5000 # make the top list of words (common words)
# embedding_dim = 64
# max_length = 600
# trunc_type = 'post'
# padding_type = 'post'
# oov_tok = '<OOV>'
# training_portion = .8

articles = []
labels = []



input_text = open('nursery_rhymes.txt').read()
new_text = input_text.lower().split('\n')
print(new_text)
fname = 'nursery_rhymes.txt'
raw_text = open(fname, 'r', encoding = 'utf-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 600
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

with open('nursery_rhymes.txt') as f:
    for line in f:
        line = line.strip('\n')
        if not line:
            break
        labels.append(line[0])
        article = line[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)

print(labels)
print(articles)



tokenizer = Tokenizer()
tokenizer.fit_on_texts(new_text)
vocab_size = len(tokenizer.word_index)+1

print(tokenizer.word_index)
print(vocab_size)

#Create sequences
input_sequences = []
for line in new_text:
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        n_gram_sequence = tokens[:i+1]
        input_sequences.append(n_gram_sequence)

#pad sequences
max_seq_len = max([len(i) for i in input_sequences])
input_seq_array = np.array(pad_sequences(input_sequences,
                                        maxlen = max_seq_len,
                                        padding='pre'))

#Create features and labels
x = input_seq_array[:, :-1]
labels = input_seq_array[:, -1]
y = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)

# build, train, and compile
model = Sequential()
model.add(Embedding(vocab_size, 60, input_length=max_seq_len-1))
model.add(Bidirectional(LSTM(60)))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam())

history = model.fit(x, y, epochs = 100, verbose = 1)


# Generate a new nursery rhyme

title = "Rhonda's Nursery Rhyme (Rhonda is the name of this model)"
seed_text = "It"
line_count = 30
line_size = 20

print(title)
print()

for i in range(line_count):
    for j in range(line_size):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list))
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    seed_text += "\n"

    
print(seed_text)