from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer, Masking, Input, Dense, concatenate
from tensorflow.keras.layers import ReLU, BatchNormalization, Attention
from tensorflow.keras.layers import BatchNormalization, Embedding, Lambda, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

class Sum(Layer):
    """
    Simple sum layer.
    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on their
    own.

    Author: Dan Guest
    https://github.com/dguest/flow-network/blob/master/SumLayer.py

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if mask is not None:
            x = x * K.cast(mask, K.dtype(x))[:,:,None]
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask):
        return None


def comp_loss(ll, n_outs):
    
    llvar = K.variable(ll)
    muon_bce = tf.keras.losses.BinaryCrossentropy()
    hit_bces = [ tf.keras.losses.BinaryCrossentropy() for i in range(n_outs - 1) ]

    def model_loss(y_true, y_pred):
            
        total_loss = muon_bce(y_true[:,0:1], y_pred[:,0:1])
            
        for ih in range(n_outs - 1):
                
            isNotMask = K.not_equal(y_true[:,ih+1:ih+2], -99) #true for all mask values
            isNotMask = K.cast(isNotMask, dtype=K.floatx())
                
            this_y_true = isNotMask*y_true[:,ih+1:ih+2]
            this_y_pred = isNotMask*y_pred[:,ih+1:ih+2]
                
            hit_loss = hit_bces[ih](this_y_true, this_y_pred)
                
            total_loss = total_loss + hit_loss*llvar
        
        return total_loss

    return model_loss
    
def muon_nn_type0(input_shape):
    
    inputs = Input(shape=(input_shape[0], input_shape[1],), name="inputs")
    
    masked = Masking( mask_value=-99, name="masking_1")(inputs)
    
    masked = BatchNormalization()(masked)
    
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_0_Dense")(masked)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_1_Dense")(phi)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_2_Dense")(phi)
    
    phi = BatchNormalization()(phi)
    
    pool = Sum()(phi)
    
    all_pool = BatchNormalization()(pool)
    
    F = Dense(20, activation='relu', name='F_0_Dense')(all_pool)
    F = Dense(10, activation='relu', name='F_1_Dense')(F)
    F = Dense(1, activation='sigmoid', name='F_Output')(F)
    
    model = Model(inputs=inputs, outputs=F)
    
    model.summary()
    
    my_loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=my_loss, optimizer='adam', metrics=[my_loss])
    
    return model

def muon_nn_type1(input_shape, ll = 0.1):
    
    inputs = Input(shape=(input_shape[0], input_shape[1],), name="inputs")
    
    masked = Masking( mask_value=-99, name="masking_1")(inputs)
    
    masked = BatchNormalization()(masked)
    
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_0_Dense")(masked)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_1_Dense")(phi)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_2_Dense")(phi)
    
    phi = BatchNormalization()(phi)
    
    rho = TimeDistributed(Dense(20,activation='relu'),name=f"Rho_0_Dense")(phi)
    rho = TimeDistributed(Dense(10,activation='relu'),name=f"Rho_1_Dense")(rho)
    rho = TimeDistributed(Dense(1,activation='sigmoid'),name=f"Rho_Output")(rho)
    all_rho = tf.keras.layers.Flatten()(rho)
    
    # to_pool = tf.keras.layers.Concatenate(axis=2)([phi, rho])
    
    pool = Sum()(phi)
    all_pool = tf.keras.layers.Concatenate(axis=1)([pool, all_rho])
    
    all_pool = BatchNormalization()(all_pool)
    
    F = Dense(20, activation='relu', name='F_0_Dense')(all_pool)
    F = Dense(10, activation='relu', name='F_1_Dense')(F)
    F = Dense(1, activation='sigmoid', name='F_Output')(F)
    
    total_output = tf.keras.layers.Concatenate(axis=1)([F, all_rho])
    
    model = Model(inputs=inputs, outputs=total_output)
    
    model.summary()
    
    my_loss = comp_loss(ll, input_shape[0]+1)
    model.compile(loss=my_loss, optimizer='adam', metrics=[my_loss])
    
    return model

def muon_nn_type2(input_shape, ll = 0.1):
    
    inputs = Input(shape=(input_shape[0], input_shape[1],), name="inputs")
    
    masked = Masking( mask_value=-99, name="masking_1")(inputs)
    
    masked = BatchNormalization()(masked)
    
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_0_Dense")(masked)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_1_Dense")(phi)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_2_Dense")(phi)
    
    phi = BatchNormalization()(phi)
    
    rho = TimeDistributed(Dense(20,activation='relu'),name=f"Rho_0_Dense")(phi)
    rho = TimeDistributed(Dense(10,activation='relu'),name=f"Rho_1_Dense")(rho)
    rho = TimeDistributed(Dense(1,activation='sigmoid'),name=f"Rho_Output")(rho)
    all_rho = tf.keras.layers.Flatten()(rho)
    
    to_pool = tf.keras.layers.Concatenate(axis=2)([phi, rho])
    to_pool = BatchNormalization()(to_pool)
    
    pool = Sum()(to_pool)
    
    all_pool = BatchNormalization()(pool)
    
    F = Dense(20, activation='relu', name='F_0_Dense')(all_pool)
    F = Dense(10, activation='relu', name='F_1_Dense')(F)
    F = Dense(1, activation='sigmoid', name='F_Output')(F)
    
    total_output = tf.keras.layers.Concatenate(axis=1)([F, all_rho])
    
    model = Model(inputs=inputs, outputs=total_output)
    
    model.summary()
    
    my_loss = comp_loss(ll, input_shape[0]+1)
    model.compile(loss=my_loss, optimizer='adam', metrics=[my_loss])
    
    return model

def muon_nn_selfatt(input_shape, ll = 0.1, attn_block=False):
    
    inputs = Input(shape=(input_shape[0], input_shape[1],), name="inputs")
    
    masked = Masking( mask_value=-99, name="masking_1")(inputs)
    
    masked = BatchNormalization()(masked)
    
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_0_Dense")(masked)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_1_Dense")(phi)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_2_Dense")(phi)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_3_Dense")(phi)
    
    # query_value_attention_seq = Attention()([phi, phi])
    query_value_attention_seq, attn_scores = Attention()([phi, phi], return_attention_scores=True)
    
    rho = TimeDistributed(Dense(50,activation='relu'),name=f"Rho_0_Dense")(query_value_attention_seq)
    rho = TimeDistributed(Dense(20,activation='relu'),name=f"Rho_1_Dense")(rho)
    rho = TimeDistributed(Dense(10,activation='relu'),name=f"Rho_2_Dense")(rho)
    rho = TimeDistributed(Dense(1,activation='sigmoid'),name=f"Rho_Output")(rho)
    all_rho = tf.keras.layers.Flatten()(rho)
    
    pool_query_value = Sum()(query_value_attention_seq)
    
    if attn_block:
        pool_attn_scores = Sum()(attn_scores)
        all_pool = tf.keras.layers.Concatenate(axis=1)([pool_query_value, pool_attn_scores])
    else:
        all_pool = pool_query_value
        
    all_pool = BatchNormalization()(all_pool)
    
    F = Dense(50, activation='relu', name='F_0_Dense')(all_pool)
    F = Dense(20, activation='relu', name='F_1_Dense')(F)
    F = Dense(10, activation='relu', name='F_2_Dense')(F)
    F = Dense(1, activation='sigmoid', name='F_Output')(F)
    
    total_output = tf.keras.layers.Concatenate(axis=1)([F, all_rho])
    
    model = Model(inputs=inputs, outputs=total_output)
    
    model.summary()
    
    my_loss = comp_loss(ll, input_shape[0]+1)
    model.compile(loss=my_loss, optimizer='adam', metrics=[my_loss])
    
    return model


def muon_nn_selfatt_reg(input_shape, ll = 0.1):
    
    inputs = Input(shape=(input_shape[0], input_shape[1],), name="inputs")
    
    masked = Masking( mask_value=-99, name="masking_1")(inputs)
    
    masked = BatchNormalization()(masked)
    
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_0_Dense")(masked)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_1_Dense")(phi)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_2_Dense")(phi)
    phi = TimeDistributed(Dense(50,activation='relu'),name=f"Phi_3_Dense")(phi)
    
    query_value_attention_seq = Attention()([phi, phi])
        
    pool = Sum()(query_value_attention_seq)
    
    all_pool = BatchNormalization()(pool)
    
    D = Dense(50, activation='relu', name='D_0_Dense')(all_pool)
    D = Dense(20, activation='relu', name='D_1_Dense')(D)
    D = Dense(10, activation='relu', name='D_2_Dense')(D)
    D = Dense(1, activation='sigmoid', name='D_Output')(D)

    Rmu = Dense(50, activation='relu', name='R_0_Dense')(all_pool)
    Rmu = Dense(20, activation='relu', name='R_1_Dense')(Rmu)
    Rmu = Dense(10, activation='relu', name='R_2_Dense')(Rmu)
    Rmu = Dense(1, activation='linear', name='R_Output')(Rmu)

    Rs = Dense(50, activation='relu', name='R_0_Dense')(all_pool)
    Rs = Dense(20, activation='relu', name='R_1_Dense')(Rmu)
    Rs = Dense(10, activation='relu', name='R_2_Dense')(Rmu)
    Rs = Dense(1, activation='linear', name='R_Output')(Rmu)
    
    total_output = tf.keras.layers.Concatenate(axis=1)([D, R])
    
    model = Model(inputs=inputs, outputs=total_output)
    
    model.summary()
    
    my_loss = comp_loss(ll, input_shape[0]+1)
    model.compile(loss=my_loss, optimizer='adam', metrics=[my_loss])
    
    return model