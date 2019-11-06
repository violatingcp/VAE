import numpy as np
#import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, LSTM, Bidirectional, RepeatVector
from keras.models import Model,Sequential
from keras import backend as K
from keras import objectives
from scipy.stats import norm

# data load
max_features = 2000
maxlen = 100
(x_tr, y_tr), (x_te, y_te) = imdb.load_data(num_words=max_features)
#x_tr, x_te = x_tr.astype('float32')/255., x_te.astype('float32')/255.
#x_tr, x_te = x_tr.reshape(x_tr.shape[0], -1), x_te.reshape(x_te.shape[0], -1)
x_tr = sequence.pad_sequences(x_tr, maxlen=maxlen)
x_te = sequence.pad_sequences(x_te, maxlen=maxlen)
print x_tr.shape[1:]
#y_tr,y_te = np.array(y_tr),np.array(y_te)

batch_size, n_epoch = 100, 100
n_hidden, z_dim = 256, 2
x = Input(shape=(x_tr.shape[1:]))
print x.shape,"1"
x1 = Embedding(max_features,128,input_length=maxlen)(x)
print x1.shape,"2"
x_encoded = LSTM(128)(x1)
x_encoded = Dropout(0.5)(x_encoded)
mu = Dense(z_dim)(x_encoded)
log_var = Dense(z_dim)(x_encoded)

# sampling function
def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps

z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])

# decoder
z_decoder1 = LSTM(128, return_sequences=True)
z_decoder2 = LSTM(128, return_sequences=True)
z_decoded = RepeatVector(maxlen)(z)
z_decoded = z_decoder1(z_decoded)
y1        = z_decoder2(z_decoded)
print y1.shape,"1",z_decoded.shape
#y         = Embedding(max_features,64,input_length=maxlen)(y1)

# loss
reconstruction_loss = objectives.binary_crossentropy(x1, y1) * x_tr.shape[1]
kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
vae_loss = reconstruction_loss + kl_loss

# build model
vae = Model(inputs=[x], outputs=[y1]) 
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

vae.fit(x_tr,
       shuffle=True,
       epochs=n_epoch,
       batch_size=batch_size,
       validation_data=(x_te, None), verbose=1)

model_json = vae.to_json()
with open("encoder_model.json", "w") as json_file:
    json_file.write(model_json)
vae.save_weights("vae_model.h5")

# build encoder
encoder = Model(inputs=[Inputs], outputs=[mu]) 
encoder.summary()
encoder_json = encoder.to_json()
with open("encoder_model.json", "w") as json_file:
    json_file.write(encoder_json)
encoder.save_weights("encoder_model.h5")

# build decoder
decoder_input = Input(shape=(z_dim,))
_h_decoded = decoder_h(_h_decoded)

_z_decoded = RepeatVector(max_features)(decoder_input)
_z_decoded = z_decoder1(_z_decoded)
_z_decoded = z_decoder2(_z_decoded)
_y         = y_decoder(_z_decoded)
generator = Model(decoder_input, _y)
generator.summary()
generator_json = generator.to_json()
with open("generator_model.json", "w") as json_file:
    json_file.write(generator_json)
generator.save_weights("generator_model.h5")


# Plot of the digit classes in the latent space
x_te_latent = encoder.predict(x_te, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_te_latent[:, 0], x_te_latent[:, 1], c=y_te)
plt.colorbar()
plt.show()

# display a 2D manifold of the digits
n = 15 # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) 
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()


