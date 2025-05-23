import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

with open("/content/drive/MyDrive/Դիպլոմային/model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("/content/drive/MyDrive/Դիպլոմային/model/latin_tokenizer.pkl", "rb") as f:
    latin_tokenizer = pickle.load(f)
with open("/content/drive/MyDrive/Դիպլոմային/model/armenian_tokenizer.pkl", "rb") as f:
    armenian_tokenizer = pickle.load(f)

latin_vocab_size = len(latin_tokenizer.word_index) + 1
armenian_vocab_size = len(armenian_tokenizer.word_index) + 1
max_latin_length = model.input_shape[0][1]
max_armenian_length = model.input_shape[1][1] + 1

latent_dim = 512

encoder_inputs = model.input[0]
encoder_outputs, state_h, state_c = model.get_layer("lstm").output
encoder_model = Model(encoder_inputs, [state_h, state_c])

decoder_inputs = Input(shape=(1,))
decoder_embedding_layer = model.get_layer("embedding_1")
decoder_embedding = decoder_embedding_layer(decoder_inputs)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm_layer = model.get_layer("lstm_1")
decoder_outputs, state_h, state_c = decoder_lstm_layer(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_dense_layer = model.get_layer("dense")
decoder_outputs = decoder_dense_layer(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def translate_text(input_text):
    input_seq = pad_sequences(latin_tokenizer.texts_to_sequences([input_text]), maxlen=max_latin_length, padding='post')
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = armenian_tokenizer.word_index['\t']
    translated_text = ""
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = armenian_tokenizer.index_word.get(sampled_token_index, "")

        if sampled_char == "\n" or len(translated_text) >= max_armenian_length:
            break

        translated_text += sampled_char
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return translated_text.strip()

while True:
    input_text = input("\nՄուտքագրեք լատինատառ տեքստը (կամ 'exit' դուրս գալու համար): ").strip()
    if input_text.lower() == "exit":
        break

    tokens = re.findall(r"\w+|[^\w\s]", input_text, re.UNICODE)
    output_text = ""

    for token in tokens:
        if re.match(r"\w+", token):  # Եթե բառ է
            clean_token = token
            translated = translate_text(clean_token)
            if token[0].isupper():
                translated = translated[0].upper() + translated[1:]

            output_text += translated
        else:
            output_text += token

        output_text += " " if token not in [",", ".", "!", "?", ":", ";"] else ""

    print("Փոխակերպում:", output_text.strip())
