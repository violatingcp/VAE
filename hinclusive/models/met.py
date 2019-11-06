from . import layers
from . import _common

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam, SGD

# Regression dense model
def build_regression(input_singletons,n_targets,loss_regression):
    h = input_singletons
    h = Dense(30, activation='relu')(h)
    h = Dense(20, activation='relu')(h)
    h = Dense(5, activation='relu')(h)
    if n_targets>1:
        out_regressor = [Dense(1, activation='linear', kernel_initializer='lecun_uniform', name='output_%i'%i)(h) for i in range(0,n_targets)]    
    else:
        out_regressor = Dense(1, name='out')(h)
    regressor = Model(inputs=[input_singletons], outputs=out_regressor, name='regressor')
    regressor.compile(optimizer=Adam(lr=0.0001),
                      loss=loss_regression,
                      metrics=['accuracy'])
    regressor.summary()
    return regressor
