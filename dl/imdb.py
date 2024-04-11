import tensorflow as tf
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


max_features = 10000 
maxlen = 100 
batch_size = 32


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) # Pad sequences to have a consistent length for the input to the RNN
x_train = pad_sequences(x_train, maxlen=maxlen) 
x_test = pad_sequences(x_test, maxlen=maxlen)


model = Sequential()
model.add(Embedding(max_features, 128)) 
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size) 
print(f'Test score:{score}')
print(f'Test accuracy: {acc}')
