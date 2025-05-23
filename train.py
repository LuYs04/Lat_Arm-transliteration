import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

file_path = "ArmLat.csv"
df = pd.read_csv(file_path)

latin_texts = df["lat"].astype(str).tolist()
armenian_texts = ["\t" + text + "\n" for text in df["arm"].astype(str).tolist()]  # Ավելացնում ենք սկիզբ ու վերջ տոկենները

latin_tokenizer = Tokenizer(char_level=True)
latin_tokenizer.fit_on_texts(latin_texts)
latin_vocab_size = len(latin_tokenizer.word_index) + 1

armenian_tokenizer = Tokenizer(char_level=True)
armenian_tokenizer.fit_on_texts(armenian_texts)
armenian_vocab_size = len(armenian_tokenizer.word_index) + 1

latin_sequences = latin_tokenizer.texts_to_sequences(latin_texts)
armenian_sequences = armenian_tokenizer.texts_to_sequences(armenian_texts)

max_latin_length = max(len(seq) for seq in latin_sequences)
max_armenian_length = max(len(seq) for seq in armenian_sequences)

latin_padded = pad_sequences(latin_sequences, maxlen=max_latin_length, padding='post')
armenian_padded = pad_sequences(armenian_sequences, maxlen=max_armenian_length, padding='post')

armenian_input = armenian_padded[:, :-1]
armenian_target = armenian_padded[:, 1:]

embedding_dim = 256
latent_dim = 512

encoder_inputs = Input(shape=(max_latin_length,))
enc_embedding = Embedding(latin_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_armenian_length - 1,))
dec_embedding = Embedding(armenian_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_embedding, initial_state=encoder_states)
decoder_dense = Dense(armenian_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    [latin_padded, armenian_input],
    np.expand_dims(armenian_target, -1),
    batch_size=128,
    epochs=10,
    validation_split=0.2
)

model.save("seq2seq_model.h5")

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("latin_tokenizer.pkl", "wb") as f:
    pickle.dump(latin_tokenizer, f)
with open("armenian_tokenizer.pkl", "wb") as f:
    pickle.dump(armenian_tokenizer, f)
