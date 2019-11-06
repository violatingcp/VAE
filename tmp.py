import numpy
from keras.datasets import imdb
from keras.models import Model,Sequential
from keras.layers import Dense
from keras.layers import Input,LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
print X_train.shape
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
print X_train.shape

#Inputs = Input(shape=(X_train.shape[1:]))
#x = Embedding(top_words,embedding_vecor_length,input_length=max_review_length)(Inputs)
#print x.shape,"1"
#x  = LSTM(100)(x)
#print x.shape,"2"
#x  = Dense(1,activation='sigmoid')(x)
#model = Model(inputs=[Inputs], outputs=[x])

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
