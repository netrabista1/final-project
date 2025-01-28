import unicodedata
import re
import math
import psutil
import time
import datetime
from io import open
import random
from random import shuffle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.cuda
from django.conf import settings
"""this line clears sys to allow for argparse to work as gradient clipper"""
import sys; sys.argv=['']; del sys

EOS_token = 1
use_cuda = True
output_file_name = "testdata.english_nepali_pairs_trim.15_vocab.50000_directions.1_layers.3_hidden.728_dropout.0.1_learningrate.1_batch.64_epochs.30"

en_word_to_index_path = os.path.join(settings.BASE_DIR, 'app',"en_word_to_index.json")
en_word_to_count_path = os.path.join(settings.BASE_DIR, 'app',"en_word_to_count.json")
ne_word_to_index_path = os.path.join(settings.BASE_DIR, 'app',"ne_word_to_index.json")
ne_word_to_count_path = os.path.join(settings.BASE_DIR, 'app',"ne_word_to_count.json")
en_index_to_word_path = os.path.join(settings.BASE_DIR, 'app',"en_index_to_word.json")
ne_index_to_word_path = os.path.join(settings.BASE_DIR, 'app',"ne_index_to_word.json")

# # Load the dictionary from the JSON file
with open(en_word_to_index_path, "r") as json_file:
    en_word_to_index = json.load(json_file)

with open(en_word_to_count_path, "r") as json_file:
    en_word_to_count = json.load(json_file)

with open(ne_word_to_index_path, "r") as json_file:
    ne_word_to_index = json.load(json_file)

with open(ne_word_to_count_path, "r") as json_file:
    ne_word_to_count = json.load(json_file)

with open(en_index_to_word_path, "r") as json_file:
    en_index_to_word = json.load(json_file)

with open(ne_index_to_word_path, "r") as json_file:
    ne_index_to_word = json.load(json_file)

def normalizeString(s):

    # special_chars = "!,@#$%^&*()_+-={}|[]:;\"'<>,.?/~`.,;?!$£€₹§©®™•"
    special_chars = "⁄‡†।!,@#$%^&*()_+-={}|[]:;\"'<>,.?/~`.,;?!$£€₹§©®™•⁇←↑→√≤≥►●、海🙂🙏𒆜𓊉꧂꧁𓊈𒆜¥"

    
    # Create a translation table for removing special characters
    translator = str.maketrans('', '', special_chars)
    
    # Remove unwanted spaces, but keep Nepali characters intact
    s = re.sub(r"\s+", " ", s)  # Replace multiple spaces with a single space
    s = s.lower()
    s = s.translate(translator)

    return s

def indexesFromSentence(sentence):
    indexes = []
    for word in sentence.split(' '):
        try:
            indexes.append(en_word_to_index[word])
        except:
            indexes.append(en_word_to_index["<UNK>"])
    return indexes

def tensorFromSentence(sentence):
    indexes = indexesFromSentence(sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes).view(-1)
    if use_cuda:
        return result.cuda()
    else:
        return result

class EncoderRNNManual(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, layers, dropout):
        super(EncoderRNNManual, self).__init__()

        # Set directions for bidirectionality
        self.directions = 2 if bidirectional else 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout_rate = dropout

        # Initialize embedding layer and dropout
        self.embedder = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Replace LSTM with custom LSTMCell
        self.lstm_cell = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=False
        )
        self.fc = nn.Linear(hidden_size * self.directions, hidden_size)

    # Actual forward code
    def forward(self, input_data, h_hidden, c_hidden):
        embedded_data = self.embedder(input_data)
        embedded_data = self.dropout(embedded_data)
        hiddens, outputs = self.lstm_cell(embedded_data, (h_hidden, c_hidden))

        return hiddens, outputs

    def create_init_hiddens(self, batch_size):
        # Create initial hidden and cell states for the encoder
        h_hidden = Variable(torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size))
        c_hidden = Variable(torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size))

        if torch.cuda.is_available():
            return h_hidden.cuda(), c_hidden.cuda()
        else:
            return h_hidden, c_hidden


class DecoderAttnManual(nn.Module):
    def __init__(self, hidden_size, output_size, layers, dropout, bidirectional):
        super(DecoderAttnManual, self).__init__()

        # Attributes and embeddings initialization
        self.directions = 2 if bidirectional else 1
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout = dropout
        self.embedder = nn.Embedding(output_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.score_learner = nn.Linear(hidden_size * self.directions, hidden_size * self.directions)
        self.lstm_cell = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=False
        )

        # Additional layers
        self.context_combiner = nn.Linear((hidden_size * self.directions) + (hidden_size * self.directions), hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size, output_size)
        self.soft = nn.Softmax(dim=1)
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, input_data, h_hidden, c_hidden, encoder_hiddens):
    # Embedding the input token
        embedded_data = self.embedder(input_data)
        embedded_data = self.dropout_layer(embedded_data)
        batch_size = embedded_data.shape[1]
    
    # Run LSTM cell
        outputs, (hiddens, c_hiddens) = self.lstm_cell(embedded_data, (h_hidden, c_hidden))

    # Compute attention scores
        prep_scores = self.score_learner(encoder_hiddens.permute(1, 0, 2))
        scores = torch.bmm(prep_scores, outputs.permute(1, 2, 0))
        attn_scores = self.soft(scores)

    # Compute context matrix and combined hidden state
        con_mat = torch.bmm(encoder_hiddens.permute(1, 2, 0), attn_scores)
        h_tilde = self.tanh(
            self.context_combiner(torch.cat((con_mat.permute(0, 2, 1), outputs.permute(1, 0, 2)), dim=2))
        )

        # Final prediction (shape: [batch_size, 1, vocab_size])
        pred = self.output(h_tilde)
    
        # Squeeze to remove the unnecessary dimension (shape: [batch_size, vocab_size])
        pred = pred.squeeze(1)
    
        # Log softmax for prediction
        pred = self.log_soft(pred)

        return pred, (hiddens, c_hiddens)

'''Returns the predicted translation of a given input sentence. Predicted
translation is trimmed to length of cutoff_length argument'''

def evaluate(encoder, decoder, sentence, cutoff_length=100):
    with torch.no_grad():
        input_variable = tensorFromSentence(sentence)
        input_variable = input_variable.view(-1, 1)
        enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(1)

        enc_hiddens, enc_outputs = encoder(input_variable, enc_h_hidden, enc_c_hidden)

        decoder_input = Variable(torch.LongTensor(1, 1).fill_(ne_word_to_index.get("SOS")).cuda()) if use_cuda \
                        else Variable(torch.LongTensor(1, 1).fill_(ne_word_to_index.get("SOS")))
        dec_h_hidden = enc_outputs[0]
        dec_c_hidden = enc_outputs[1]

        decoded_words = []

        # Print type of trim (or cutoff_length) to debug
        # print(f"Type of trim: {type(cutoff_length)}")  # Check the type of cutoff_length

        for di in range(cutoff_length):
            pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)

            topv, topi = pred.topk(1, dim=1)
            ni = topi.item()
            ni = str(ni)
            if ni == str(ne_word_to_index.get("EOS")):
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(ne_index_to_word[ni])

            decoder_input = Variable(torch.LongTensor(1, 1).fill_(int(ni)).cuda()) if use_cuda \
                            else Variable(torch.LongTensor(1, 1).fill_(int(ni)))
            dec_h_hidden = dec_outputs[0]
            dec_c_hidden = dec_outputs[1]

        output_sentence = ' '.join(decoded_words)

        return output_sentence

"""create the Encoder"""
encoder = EncoderRNNManual(38604, 728, layers=3,
                     dropout=0.1, bidirectional=False)

"""create the Decoder"""
decoder = DecoderAttnManual(728, 31228, layers=3,
                      dropout=0.1, bidirectional=False)

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# encoder.load_state_dict(torch.load(output_file_name+'_enc_weights.pt'))
# decoder.load_state_dict(torch.load(output_file_name+'_dec_weights.pt'))

enc_weights_path = os.path.join(settings.BASE_DIR,'app', f"{output_file_name}_enc_weights.pt")
dec_weights_path = os.path.join(settings.BASE_DIR,'app', f"{output_file_name}_dec_weights.pt")

# Load the state dictionaries
encoder.load_state_dict(torch.load(enc_weights_path))
decoder.load_state_dict(torch.load(dec_weights_path))
encoder.eval()
decoder.eval()
def translate_sentence(sentence):
    # Normalize the input sentence
    outside_sent = normalizeString(sentence)
    
    # Evaluate the sentence using the encoder and decoder
    nep = evaluate(encoder, decoder, outside_sent, cutoff_length=20)
    
    # Split the translated sentence into tokens
    nep_tokens = nep.split(" ")
    
    # Remove the <EOS> token from the tokens, if present
    nep_tokens = [token for token in nep_tokens if token != "<EOS>"]
    
    # Remove consecutive duplicate tokens
    filtered_tokens = []
    for token in nep_tokens:
        if not filtered_tokens or token != filtered_tokens[-1]:
            filtered_tokens.append(token)
    
    # Join the tokens back into a single string
    nep = " ".join(filtered_tokens).strip()
    
    # Check if the input English sentence has only one word
    word_count = 0
    for word in outside_sent.split(' '):
        if word.strip():  # Avoid counting empty strings
            word_count += 1
    
    # If the English sentence has only one word, return the first token of the Nepali translation
    if word_count == 1:
        return filtered_tokens[0] if filtered_tokens else ""
    
    return nep

