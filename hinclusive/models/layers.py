import numpy as np
import keras.backend as K
from keras.layers import Layer
import tensorflow as tf
from . import gaussian

"""Flips the sign of the incoming gradient during training."""
def _reverse(x, scale = 1):
    # first make this available to tensorflow                                                                                                                                                                
    if hasattr(_reverse, 'N'):
        _reverse.N += 1
    else:
        _reverse.N = 1
    name = 'reverse%i'%_reverse.N

    @tf.RegisterGradient(name)
    def f(op, g):
        # this is the actual tensorflow op                                                                                                                                                                     
        return [scale * tf.negative(g)]

    graph = K.get_session().graph
    with graph.gradient_override_map({'Identity':name}):
        ret = tf.identity(x)

    return ret

class GradReverseLayer(Layer):
    def __init__(self, scale = 1, **kwargs):
        self.scale = scale
        super(GradReverseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []
        super(GradReverseLayer, self).build(input_shape)

    def call(self, x):
        return _reverse(x, self.scale)

    def get_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super(GradReverseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Adversary(object):
    def __init__(self, n_output_bins, n_outputs=1, scale=1):
        self.scale = scale
        self.n_output_bins = n_output_bins
        self.n_outputs = n_outputs
        self._outputs = None
        self._dense = []

    def __call__(self, inputs):
        self._reverse = GradReverseLayer(self.scale, name='u_grl')(inputs)

        n_outputs = self.n_outputs
        self._outputs = None
        self._dense = []

    def __call__(self, inputs):
        self._reverse = GradReverseLayer(self.scale, name='u_grl')(inputs)

        n_outputs = self.n_outputs
        self._dense.append( [Dense(5, activation='tanh')(self._reverse) for _ in xrange(n_outputs)] )
        self._dense.append( [Dense(10, activation='tanh')(d) for d in self._dense[-1]] )
        self._dense.append( [Dense(10, activation='tanh')(d) for d in self._dense[-1]] )
        self._dense.append( [Dense(10, activation='tanh')(d) for d in self._dense[-1]] )
        if self.n_output_bins > 1:
            if n_outputs == 1:
                self._outputs = [Dense(self.n_output_bins, activation='softmax', name='adv')(d)
                                 for d in self._dense[-1]]
            else:
                self._outputs = [Dense(self.n_output_bins, activation='softmax', name='adv%i'%i)(d)
                                 for i,d in enumerate(self._dense[-1])]
        else:
            if n_outputs == 1:
                self._outputs = [Dense(1, activation='linear', name='adv')(d) for d in self._dense[-1]]
            else:
                self._outputs = [Dense(1, activation='linear', name='adv%i'%i)(d) for i,d in enumerate(self._dense[-1])]
        return self._outputs


class PosteriorLayer (Layer):

    def __init__ (self, nb_gmm, **kwargs):
        """
        Custom layer, modelling the posterior probability distribution for the jet mass using a gaussian mixture model (GMM)
        """
        # Base class constructor
        super(PosteriorLayer, self).__init__(**kwargs)

        # Member variable(s) 
        self.nb_gmm = nb_gmm
        pass

    def call (self, x, mask=None):
        """Main call-method of the layer.                                                                                                                                                                
        The GMM needs to be implemented (1) within this method and (2) using                                                                                                                                 
        Keras backend functions in order for the error back-propagation to work                                                                                                                             
        properly.                                                                                                                                                                                             
        """

        # Unpack list of inputs                                                                                                                                                                                
        coeffs, means, widths, m = x

        # Compute the pdf from the GMM                                                                                                                                                                         
        pdf = gaussian.GMM(m[:,0], coeffs, means, widths, self.nb_gmm)

        return K.flatten(pdf)

    def compute_output_shape (self, input_shape):
        return (input_shape[0][0], 1)

    pass
