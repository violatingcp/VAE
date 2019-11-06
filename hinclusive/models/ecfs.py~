from . import layers
from . import _common 
from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras import objectives

# Classifier dense model
def build_classifier(input_singletons,input_mass,num_vars,loss_classifier):
    N = num_vars
    h = input_singletons
    h = Dense(30, activation='relu')(h)
    h = Dense(20, activation='relu')(h)
    h = Dense(5, activation='relu')(h)
    out_classifier = Dense(1, activation='sigmoid', name='out')(h)
    inputs=[input_singletons]
    classifier = Model(inputs=inputs, outputs=[out_classifier])
    classifier.compile(optimizer=Adam(lr=0.0001),
                       loss=loss_classifier,
                       metrics=['accuracy'])
    print('########### CLASSIFIER ############')
    classifier.summary()
    print('###################################')

    return classifier


# Classifier dense model
def build_vae(input_singletons,input_mass,num_vars,loss_classifier,batch_size,in_shape):
    z_dim = 2
    N = num_vars
    h = input_singletons
    h = Dense(30, activation='relu')(h)
    h = Dense(20, activation='relu')(h)
    h = Dense(5, activation='relu')(h)
    mu      = Dense(z_dim)(h)
    log_var = Dense(z_dim)(h)
    
    # sampling function
    def sampling(args):
        mu, log_var = args
        eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
        return mu + K.exp(log_var) * eps

    z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])
    z = Dense(5, activation='relu')(z)
    z = Dense(20, activation='relu')(z)
    z = Dense(30, activation='relu')(z)
    out_classifier  = Dense(in_shape, activation='sigmoid')(z)
    inputs=[input_singletons]
    classifier = Model(inputs=inputs, outputs=[out_classifier])
    print in_shape,batch_size,"!!!"
    reconstruction_loss = objectives.binary_crossentropy(input_singletons, out_classifier) * in_shape
    kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
    vae_loss = reconstruction_loss + kl_loss
    classifier.add_loss(vae_loss)
    classifier.compile(optimizer='rmsprop')##optimizer=Adam(lr=0.0001),
    print('########### CLASSIFIER ############')
    classifier.summary()
    print('###################################')

    return classifier

# Adversary mass
def adversary_model (nb_gmm):

    # Input(s)
    i = Input(shape=(1,))
    m = Input(shape=(1,))
    
    # Hidden layer(s)
    x = Dense(4,  activation='relu')(i)
    x = Dense(8,  activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    # Gaussian mixture model (GMM) components
    coeffs = Dense(nb_gmm, activation='softmax') (x)  # GMM coefficients sum to one
    means  = Dense(nb_gmm, activation='sigmoid') (x)  # Means are on [0, 1]
    widths = Dense(nb_gmm, activation='softplus')(x)  # Widths are positive
    
    # Posterior p.d.f.
    pdf = layers.PosteriorLayer(nb_gmm)([coeffs, means, widths, m])

    # Build model
    adversary = Model(inputs=[i,m], outputs=pdf, name='adversary')
    adversary.compile(optimizer='adam', loss=_common.KL)
    print('########### ADVERSARY ############')
    adversary.summary()
    print('###################################')
    
    return adversary

# Combined method with mass
def combined_model (clf, adv, lambda_reg, lr_ratio,loss_classifier='binary_crossentropy',loss_weights = [1.0E-08, 1.]):

    # Classifier
    input_clf  = Input(shape=clf.layers[0].input_shape[1:])
    input_m    = Input(shape=(1,))
    output_clf = clf(input_clf)

    # Connect with gradient reversal layer
    gradient_reversal = layers.GradReverseLayer(lambda_reg * lr_ratio)(output_clf)

    # Adversary
    output_adv = adv([gradient_reversal, input_m])

    # Build model
    combined = Model(inputs=[input_clf, input_m], outputs=[output_clf, output_adv], name='combined')
    combined.compile(optimizer='adam', loss=[loss_classifier, _common.KL], loss_weights=loss_weights)
    print('########### COMBINED ############')
    combined.summary()
    print('###################################')
    return combined

# Adversary from h decay
def build_adversary_decay(Y_adv_train, frozen_classifier,lossfunction='binary_crossentropy'):
    LWR=0.001
    inputs= frozen_classifier.inputs
    n_components = Y_adv_train.shape[1] # 7 decays bb WW tt cc glu glu ZZ other
    adv   = frozen_classifier.outputs[0]
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(n_components, activation="sigmoid")(adv)
    advM  = Model(inputs=inputs, outputs=[adv], 
                  name='adv_base')
    DRf = Model(inputs=inputs,
                outputs=[frozen_classifier(inputs), advM(inputs)], 
                name='adv_full')
    opt_DRf = SGD(momentum=0.0)
    DRf.compile(loss=[lossfunction,_common.conditional_loss_function],loss_weights=[1.0,-1./LWR], optimizer=opt_DRf)
    DfR = Model(inputs=inputs, outputs=[advM(inputs)], name='adv_back') #advback
    opt_DfR = SGD(momentum=0.0)
    DfR.compile(loss=['categorical_crossentropy'], optimizer=opt_DfR)
    
    print('########### Adv Decay (base) ############')
    advM.summary()
    print('###################################')
    print('########### Adv Decay (combined) ############')
    DRf.summary()
    print('###################################')
    print('########### Adv Decay (back) ############')
    DfR.summary()
    print('###################################')
    return advM,DRf,DfR

# adversary h decay
def adversary_decay_only(Y_adv_train,clf):

    # Input(s)
    n_components = Y_adv_train.shape[1]
    i = Input(shape=(1,))

    # decay adv
    adv   = Dense(20, activation="relu")(i)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(n_components, activation="sigmoid")(adv)
    
    # Build model
    #adversary = Model(inputs=inputs, outputs=adv, name='adversary_decay')
    adversary = Model(inputs=i, outputs=adv, name='adversary_decay')
    adversary.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
    #opt_DfR = SGD(momentum=0.0)
    #adversary.compile(optimizer=opt_DfR, loss=['categorical_crossentropy'], metrics=['accuracy'])
    print('########### ADVERSARY DECAY ############')
    adversary.summary()
    print('###################################')
    
    return adversary
                                        
# combined with decay only
def combined_model_decay(clf, adv_decay, lambda_reg, lr_ratio,loss_classifier='binary_crossentropy',loss_weights = [1.0E-08, 1.]):
    
    # Classifier
    input_clf  = Input(shape=clf.layers[0].input_shape[1:])
    output_clf = clf(input_clf)
    
    # Connect with gradient reversal layer
    gradient_reversal = layers.GradReverseLayer(lambda_reg * lr_ratio)(output_clf)
        
    # Adversary
    output_adv = adv_decay([gradient_reversal])

    # Build model
    combined_decayadv = Model(inputs=[input_clf],
                              outputs=[output_clf, output_adv], name='combined_decayonly')
    combined_decayadv.compile(optimizer='adam', loss=[loss_classifier, _common.conditional_loss_function], loss_weights=loss_weights)
    print('########### Adv Decay only (combined) ############')
    combined_decayadv.summary()
    print('###################################')
    return combined_decayadv

# adversary mass and decay
def adversary_mass_decay(nb_gmm,Y_adv_train,loss_weights = [1., 1000.]):

    # Input(s)
    n_components = Y_adv_train.shape[1]
    i = Input(shape=(1,))
    m = Input(shape=(1,))
    #i_decay = Input(shape=(n_components,))
    i_decay = Input(shape=(1,))
    
    # Hidden layer(s)
    x = Dense(4,  activation='relu')(i)
    x = Dense(8,  activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    
    # Gaussian mixture model (GMM) components
    coeffs = Dense(nb_gmm, activation='softmax') (x)  # GMM coefficients sum to one
    means  = Dense(nb_gmm, activation='sigmoid') (x)  # Means are on [0, 1]
    widths = Dense(nb_gmm, activation='softplus')(x)  # Widths are positive
    
    # Posterior p.d.f.
    pdf = layers.PosteriorLayer(nb_gmm)([coeffs, means, widths, m])

    # decay adv
    #adv   = frozen_classifier.outputs[0]
    #adv   = Dense(20, activation="relu")(i_decay)
    adv   = Dense(20, activation="relu")(i)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(n_components, activation="sigmoid")(adv)
    
    # Build model
    #adversary = Model(inputs=[i,m,i_decay], outputs=[pdf,adv], name='adversary')
    adversary = Model(inputs=[i,m], outputs=[pdf,adv], name='adversary_mass_decay')
    #adversary = Model(inputs=i, outputs=adv, name='adversary_decay')
    #opt_DfR = SGD(momentum=0.0)
    #adversary.compile(optimizer=opt_DfR, loss=[_common.KL,'categorical_crossentropy'], loss_weights=loss_weights, metrics=['accuracy'])
    adversary.compile(optimizer=Adam(lr=0.001), loss=[_common.KL,'categorical_crossentropy'], loss_weights=loss_weights, metrics=['accuracy'])
    print('########### ADVERSARY WITH MASS AND DECAY ############')
    adversary.summary()
    print('###################################')
    
    return adversary

# Combined with decay
def combined_model_mass_decay(clf, adv_decay, lambda_reg, lr_ratio,loss_classifier='binary_crossentropy',loss_weights = [1.0E-08, 1., 1.]):

    # Classifier
    input_clf  = Input(shape=clf.layers[0].input_shape[1:])
    input_m    = Input(shape=(1,))
    output_clf = clf(input_clf)

    # Connect with gradient reversal layer
    gradient_reversal = layers.GradReverseLayer(lambda_reg * lr_ratio)(output_clf)
    
    # Adversary
    #output_adv = adv_decay([gradient_reversal, input_m, gradient_reversal])
    output_adv = adv_decay([gradient_reversal, input_m])
    
    # Build model
    combined_decayadv = Model(inputs=[input_clf, input_m], 
                              outputs=[output_clf] + output_adv, name='combined_mass_decay')
    combined_decayadv.compile(optimizer='adam', loss=[loss_classifier, _common.KL,_common.conditional_loss_function], loss_weights=loss_weights)
    print('########### Adv Decay and Mass (combined) ############')
    combined_decayadv.summary()
    print('###################################')
    return combined_decayadv

# Old adversary model
# lr used to be 0.00025
def build_adversary(frozen_classifier, loss, scale, w_clf, w_adv, lr=0.0001, loss_classifier='binary_crossentropy'):
    inputs= frozen_classifier.inputs
    y_hat = frozen_classifier.outputs[0]
    kin_hats = layers.Adversary(n_decorr_bins, n_outputs=1, scale=scale)(y_hat)
    adversary = Model(inputs=inputs,
                      outputs=[y_hat]+kin_hats)       
    adversary.compile(optimizer=Adam(lr=lr),
                      loss=[loss_classifier]+[loss for _ in kin_hats],
                      loss_weights=[w_clf]+[w_adv for _ in kin_hats])
    print('########### ADVERSARY ############')
    adversary.summary()
    print('###################################')

    return adversary
