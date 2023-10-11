#code NLP---------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import streamlit as st
import random

# text processing
from nltk.tokenize import word_tokenize, sent_tokenize


import pandas as pd

df = pd.read_csv("clean_data_new.csv")
def tokenizing_each_sentence(df,column):
    list_of_words = []
    for i in df[column]:
        list_of_words.append(word_tokenize(i))
    return list_of_words

df['preprocessed'] = df['preprocessed'].astype(str)
tests = tokenizing_each_sentence(df,'preprocessed')
#--------------------------------------------------------------------------------------------

st.title('Autocomplete with Neural Network')
st.write('Kelompok 10 NLP')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

@st.cache_resource
def train_model(df, tests):
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['preprocessed'])  # Fit on the preprocessed text
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences and labels
    input_sequences = []
    for sentence in tests:
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

    # Create predictors and labels
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = to_categorical(label, num_classes=total_words)

    # Build and train the model
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(predictors, label, epochs=30, verbose=1)

    return model, tokenizer, max_sequence_length


model, tokenizer, max_sequence_length = train_model(df, tests)

# Function to predict next word
@st.cache_resource
def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    return list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(predicted)]

# Input teks
user_input = st.text_input('Masukkan teks anda', 'i like')    
#previous_tokens = user_input.split()

seed_text = user_input
predicted_text = seed_text

pred_kata = []
for _ in range(3):
    next_word = predict_next_word(seed_text)
    pred_kata.append(next_word)
    seed_text = " ".join(pred_kata)  
    
autocomplete = [user_input] + pred_kata

st.subheader("Neural Network")
st.write('Prediksi Kata:',' '.join(autocomplete))

st.link_button("N-gram", "http://localhost:8502")