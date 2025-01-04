import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Fungsi untuk memuat dataset
def load_data(filepath):
    # Baca file movie_lines.txt
    lines = open(filepath, encoding='utf-8', errors='ignore').read().split('\n')
    # Baca file movie_conversations.txt
    conversations = open('cornell_data\cornell_movie_dialogs_corpus\cornell movie-dialogs corpus\movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    
    # Buat dictionary id2line
    id2line = {}
    for line in lines:
        parts = line.split(' +++$+++ ')
        if len(parts) == 5:
            id2line[parts[0]] = parts[4]
    
    # Ambil percakapan
    conversation_ids = []
    for conv in conversations[:-1]:
        parts = conv.split(' +++$+++ ')[-1]
        conversation_ids.append(eval(parts))
    
    # Buat daftar pertanyaan dan jawaban
    questions, answers = [], []
    for conv in conversation_ids:
        for i in range(len(conv) - 1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i + 1]])
    
    return questions, answers

# Fungsi preprocessing teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# Fungsi konversi teks menjadi token
def text_to_sequences(texts, word2idx):
    sequences = []
    for text in texts:
        tokens = text.split()
        seq = [word2idx[token] for token in tokens if token in word2idx]
        sequences.append(seq)
    return sequences

# Load dataset
print("Memuat dataset...")
questions, answers = load_data('cornell_data\cornell_movie_dialogs_corpus\cornell movie-dialogs corpus\movie_lines.txt')

# Preprocessing teks
print("Preprocessing teks...")
questions = [clean_text(q) for q in questions]
answers = [clean_text(a) for a in answers]
answers = ["<START> " + a + " <END>" for a in answers]

# Membuat tokenizer
all_words = set(word for sentence in questions + answers for word in sentence.split())
word2idx = {word: i + 1 for i, word in enumerate(all_words)}
idx2word = {i: word for word, i in word2idx.items()}

# Konversi teks ke token
input_sequences = text_to_sequences(questions, word2idx)
output_sequences = text_to_sequences(answers, word2idx)

# Padding sequence
max_length = max(len(seq) for seq in input_sequences + output_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_length, padding='post')

# Membuat data target untuk decoder
decoder_target_data = np.array([to_categorical(seq, num_classes=len(word2idx) + 1) for seq in output_sequences])

# Parameter model
embedding_dim = 256
hidden_units = 512

# Encoder
encoder_inputs = Input(shape=(max_length,))
encoder_embedding = Embedding(len(word2idx) + 1, embedding_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(hidden_units, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_length,))
decoder_embedding = Embedding(len(word2idx) + 1, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(word2idx) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compile model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Latih model
batch_size = 64
epochs = 20
print("Melatih model...")
model.fit(
    [input_sequences, output_sequences],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

# Simpan model
model.save('chatbot_model.h5')

# Encoder model untuk inferensi
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model untuk inferensi
decoder_state_input_h = Input(shape=(hidden_units,))
decoder_state_input_c = Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_infer = decoder_embedding(decoder_inputs)
decoder_lstm_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding_infer, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_lstm_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Fungsi untuk prediksi respons
def predict_response(input_text):
    input_seq = pad_sequences([text_to_sequences([input_text], word2idx)[0]], maxlen=max_length, padding='post')
    states_value = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx['<START>']
    
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx2word.get(sampled_token_index, '')
        
        if sampled_word == '<END>' or len(decoded_sentence) > max_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    
    return decoded_sentence.strip()

# Chatbot interface
print("Mulai chatting dengan bot! Ketik 'quit' untuk keluar.")
while True:
    input_text = input("You: ")
    if input_text.lower() == 'quit':
        print("Bot: Sampai jumpa!")
        break
    response = predict_response(clean_text(input_text))
    print("Bot:", response)
