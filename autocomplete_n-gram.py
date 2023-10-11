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

df['cleaned'] = df['cleaned'].astype(str)
tests = tokenizing_each_sentence(df,'cleaned')
#--------------------------------------------------------------------------------------------

st.title('Autocomplete with N-gram Probability')
st.write('Kelompok 10 NLP')

def count_words(tokenized_sentences):
    word_counts = {}
    
    for sentence in tokenized_sentences: 
        for token in sentence:
            word_counts[token] = word_counts.get(token, 0) + 1
    
    return word_counts

def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    closed_vocab = []
    word_counts = count_words(tokenized_sentences)

    for word, count in word_counts.items():
        if count >= count_threshold:
            closed_vocab.append(word)
    
    return closed_vocab

vocabulary = get_words_with_nplus_frequency(tests, count_threshold=1)

def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + k * vocabulary_size
    
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    
    numerator = n_plus1_gram_count + k
    probability = numerator / denominator  
    
    return probability
def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    
    vocabulary = vocabulary + ['<e>', '<unk>']
    vocabulary_size = len(vocabulary)
    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, 
                                        n_gram_counts, n_plus1_gram_counts, 
                                        vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    n = len(list(n_gram_counts.keys())[0])
    previous_n_gram = previous_tokens[-n:]
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)
    suggestion = None
    max_prob = 0
    
    for word, prob in probabilities.items():
        if start_with:
            if not word.startswith(start_with):
                continue
        
        if prob > max_prob:
            suggestion = word
            max_prob = prob
    
    return suggestion, max_prob
def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    n_grams = {}
    
    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]
        sentence = tuple(sentence)
        
        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i : i + n]
            n_grams[n_gram] = n_grams.get(n_gram, 0) + 1
    
    return n_grams

def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

n_gram_counts_list = []
for n in range(1, 6):
    n_model_counts = count_n_grams(tests, n)
    n_gram_counts_list.append(n_model_counts)

# Input teks
user_input = st.text_input('Masukkan teks anda', 'i like')    
previous_tokens = user_input.split()

suggest = get_suggestions(previous_tokens,n_gram_counts_list,vocabulary, k=1.0)

highest_probability_word = suggest[0][0]
if highest_probability_word == "<e>":
    highest_probability_word = "."
autocomplete = previous_tokens + [highest_probability_word]

st.subheader("N-gram probability")
st.write('Prediksi Kata:',' '.join(autocomplete))

st.link_button("LSTM", "http://localhost:8501")