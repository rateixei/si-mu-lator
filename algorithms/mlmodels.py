from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer, Masking, Input, Dense, concatenate, Add
from tensorflow.keras.layers import ReLU, BatchNormalization, Attention
from tensorflow.keras.layers import BatchNormalization, Embedding, Lambda, TimeDistributed
from tensorflow.keras.layers import LSTM, GRU, Conv1D, GlobalMaxPooling1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm

import tensorflow as tf

import sys

class Sum(Layer):
    """
    Simple sum layer. Needed for deep sets model.
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

def muon_and_hit_loss(ll, n_outs):
    """
    Custom loss that adds an event-based binary classifier loss (muon vs no muon)
    and many hit-based binary classifier losses (muon hit vs noise hit).
    The two components are importance weighted by input ll.

    """
    
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

      
def regr_loss(do_angle=0, pen_type=0, pen_x=True, pen_a=True, bkg_pen_x=False, bkg_pen_a=False ):
    """
    Custom loss that adds an event-based binary classifier loss (muon vs no muon)
    and many hit-based binary classifier losses (muon hit vs noise hit).
    The two components are importance weighted by input ll.

    """
    
    half = tf.constant(0.5, dtype=tf.float32)
    max_val = tf.constant(1, dtype=tf.float32)
    min_val = tf.constant(-1, dtype=tf.float32)
    pow_val = tf.constant(2, dtype=tf.float32)
    
    
    reduct = tf.constant(1./100., dtype=tf.float32)
    
    
    regrloss_x = tf.keras.losses.MeanAbsolutePercentageError(reduction='sum_over_batch_size')
    regrloss_a = tf.keras.losses.MeanAbsolutePercentageError(reduction='sum_over_batch_size')
    do_ang_const = tf.constant(1., dtype=tf.float32)
    do_x_const = tf.constant(1., dtype=tf.float32)
    do_clas_const = tf.constant(1., dtype=tf.float32)
    
    def penalty_func(pred, bound_val):
        if pen_type == 2:
            return tf.math.abs( pred - bound_val ) + tf.math.pow( pred - bound_val, pow_val )
        elif pen_type == 1:
            return tf.math.abs( pred - bound_val )
        else:
            return tf.math.pow( pred - bound_val, pow_val )

    def model_loss(y_true, y_pred):
        
        # tf.print(y_true.shape, y_pred.shape, y_pred[:,0].shape, output_stream=sys.stdout)
        regrloss_val_x = regrloss_x(y_true[:,1], y_pred[:,0])
        # return regrloss_val_x

        #regrloss_val_a = regrloss_a(y_true[:,2], y_pred[:,1])
        
        regr_penalty_x1 = penalty_func(y_pred[:,0], max_val) * tf.cast( tf.math.greater(y_pred[:,0], max_val), dtype=tf.float32 )
        regr_penalty_x2 = penalty_func(y_pred[:,0], min_val) * tf.cast( tf.math.less(   y_pred[:,0], min_val), dtype=tf.float32 ) 
        regr_penalty_x = regr_penalty_x1 + regr_penalty_x2
        
        # regr_penalty_a1 = penalty_func(y_pred[:,1], max_val) * tf.cast( tf.math.greater(y_pred[:,1], max_val), dtype=tf.float32 )
        #regr_penalty_a2 = penalty_func(y_pred[:,1], min_val) * tf.cast( tf.math.less(   y_pred[:,1], min_val), dtype=tf.float32 ) 
        #regr_penalty_a = regr_penalty_a1 + regr_penalty_a2
        
        loss_x = tf.cast(y_true[:,0], dtype=tf.float32)*regrloss_val_x
        if pen_x:
            loss_x += regr_penalty_x
        if bkg_pen_x:
            loss_x += (1. - tf.cast(y_true[:,0], dtype=tf.float32))*penalty_func(y_pred[:,0], 0.)
        
        #loss_a = tf.cast(y_true[:,0], dtype=tf.float32)*regrloss_val_a 
        #if pen_a:
        #    loss_a += regr_penalty_a
        #if bkg_pen_a:
        #    loss_a += (1. - tf.cast(y_true[:,0], dtype=tf.float32))*penalty_func(y_pred[:,1], 0.)
        
        
        #loss_regr = do_x_const*loss_x + do_ang_const*loss_a
        
        #return reduct*loss_regr
        return reduct*loss_x

    return model_loss


def class_and_regr_loss(ll, do_angle=0, pen_type=0, pen_x=True, pen_a=True, bkg_pen_x=False, bkg_pen_a=False, linearized=False):
    """
    Custom loss that adds an event-based binary classifier loss (muon vs no muon)
    and many hit-based binary classifier losses (muon hit vs noise hit).
    The two components are importance weighted by input ll.

    """
    
    llvar = tf.constant(ll, dtype=tf.float32)
    half = tf.constant(0.5, dtype=tf.float32)
    max_val = tf.constant(1, dtype=tf.float32)
    min_val = tf.constant(-1, dtype=tf.float32)
    pow_val = tf.constant(2, dtype=tf.float32)
    
    
    reduct = tf.constant(1./100., dtype=tf.float32)
    
    
    clasloss = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size')
    regrloss_x = tf.keras.losses.MeanAbsolutePercentageError(reduction='sum_over_batch_size')
    regrloss_a = tf.keras.losses.MeanAbsolutePercentageError(reduction='sum_over_batch_size')
    indy = 0
    indx = 1
    inda = 2 if do_angle else 1
    do_ang_const = tf.constant(1., dtype=tf.float32)
    do_x_const = tf.constant(1., dtype=tf.float32)
    do_clas_const = tf.constant(1., dtype=tf.float32)
    
    def penalty_func(pred, bound_val):
        if pen_type == 2:
            return tf.math.abs( pred - bound_val ) + tf.math.pow( pred - bound_val, pow_val )
        elif pen_type == 1:
            return tf.math.abs( pred - bound_val )
        else:
            return tf.math.pow( pred - bound_val, pow_val )

    def model_loss(y_true, y_pred):

        y_cpred = tf.keras.activations.sigmoid(y_pred[:,indy]) if linearized else y_pred[:,indy]
        
        clasloss_val = clasloss(y_true[:,indy], y_cpred)

        regrloss_val_x = regrloss_x(y_true[:,indx]*y_true[:,indy], y_pred[:,indx]*y_true[:,indy])
        regrloss_val_a = regrloss_a(y_true[:,inda]*y_true[:,indy], y_pred[:,inda]*y_true[:,indy])
        
        regr_penalty_x1 = penalty_func(y_pred[:,indx], max_val) * tf.cast( tf.math.greater(y_pred[:,indx], max_val), dtype=tf.float32 )
        regr_penalty_x2 = penalty_func(y_pred[:,indx], min_val) * tf.cast( tf.math.less(y_pred[:,indx], min_val), dtype=tf.float32 ) 
        regr_penalty_x  = regr_penalty_x1 + regr_penalty_x2
        
        regr_penalty_a1 = penalty_func(y_pred[:,inda], max_val) * tf.cast( tf.math.greater(y_pred[:,inda], max_val), dtype=tf.float32 )
        regr_penalty_a2 = penalty_func(y_pred[:,inda], min_val) * tf.cast( tf.math.less(y_pred[:,inda], min_val), dtype=tf.float32 ) 
        regr_penalty_a = regr_penalty_a1 + regr_penalty_a2
        
        loss_x = tf.cast(y_true[:,indy], dtype=tf.float32)*regrloss_val_x
        if pen_x:
            loss_x += regr_penalty_x
        if bkg_pen_x:
            loss_x += (1. - tf.cast(y_true[:,indy], dtype=tf.float32))*penalty_func(y_pred[:,indx], 0.)
        
        loss_a = tf.cast(y_true[:,indy], dtype=tf.float32)*regrloss_val_a 
        if pen_a:
            loss_a += regr_penalty_a
        if bkg_pen_a:
            loss_a += (1. - tf.cast(y_true[:,indy], dtype=tf.float32))*penalty_func(y_pred[:,inda], 0.)
        
        
        loss_regr = do_x_const*loss_x + do_ang_const*loss_a
        
        total_loss =  do_clas_const*clasloss_val + reduct*llvar*loss_regr

        return total_loss

    return model_loss


def class_and_quantregr_loss(ll, do_angle=0):
    """
    Custom loss that adds an event-based binary classifier loss (muon vs no muon)
    and many hit-based binary classifier losses (muon hit vs noise hit).
    The two components are importance weighted by input ll.

    """
    
    def tilted_loss(q,y,f):
        e = (y-f)
        return tf.math.reduce_mean(tf.math.maximum(q*e, (q-1)*e), axis=-1)
    
    llvar = tf.constant(ll, dtype=tf.float32)
    half = tf.constant(0.5, dtype=tf.float32)
    max_val = tf.constant(1, dtype=tf.float32)
    min_val = tf.constant(-1, dtype=tf.float32)
    pow_val = tf.constant(2, dtype=tf.float32)
    clasloss = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size')
    regrloss_x = tf.keras.losses.MeanAbsoluteError(reduction='sum_over_batch_size')
    regrloss_a = tf.keras.losses.MeanAbsoluteError(reduction='sum_over_batch_size')
    indy = 0
    indx = 1
    indqx = 2
    inda = 3 if do_angle else 1
    indqa = 4 if do_angle else 1


    def model_loss(y_true, y_pred):
        clasloss_val = clasloss(y_true[:,indy], y_pred[:,indy])
        
        regrloss_val_x = regrloss_x(y_true[:,indx], y_pred[:,indx])
        regrloss_val_qx = tilted_loss(0.84,y_true[:,indx], y_pred[:,indx])
        
        regrloss_val_a = regrloss_a(y_true[:,inda], y_pred[:,inda])
        regrloss_val_qa = tilted_loss(0.84,y_true[:,inda], y_pred[:,inda])
        
        regr_penalty_1 = tf.math.pow(y_pred[:,indx] - max_val, pow_val)*tf.cast( tf.math.greater(y_pred[:,indx], max_val), dtype=tf.float32 )
        regr_penalty_2 = tf.math.pow(y_pred[:,indx] - min_val, pow_val)*tf.cast( tf.math.less(y_pred[:,indx], min_val), dtype=tf.float32 ) 
        regr_penalty_x = regr_penalty_1 + regr_penalty_2
        
        loss_x = tf.cast(y_true[:,indy], dtype=tf.float32)*(regrloss_val_x + regrloss_val_qx) + regr_penalty_x 
        loss_a = tf.cast(y_true[:,indy], dtype=tf.float32)*(regrloss_val_a + regrloss_val_qa)
        
        loss_regr = loss_x + do_angle*loss_a
        
        total_loss =  clasloss_val + llvar*loss_regr

        return total_loss

    return model_loss
    
def model_deep_set_muon(input_shape, 
        phi_layers=[50,50,50], 
        F_layers=[20,10], 
        batchnorm=True, mask_v=0, 
        add_selfattention=False,
        do_reg_out=0,
        masking=True):
    """
    Implementation of deep sets model with muon outputs only.
    phi_layers correspond to the architecture of the hit-level 
    feature extraction network.
    F_layers correspond to the architecture of the event-level 
    classifier network.
    mask_v sets input values to be masked.
    add_selfattention adds a self-attention block.
    If do_reg_out==0, the network is trained as a binary classifier.
    If do_reg_out>0, the network is trained as a regressor with 
    do_reg_out outputs.
    
    """


    if len(phi_layers) < 1:
        print("Invalid phi_layers", phi_layers)
        return -1
    
    if len(F_layers) < 1:
        print("Invalid F_layers", F_layers)
        return -1
    
    inputs = Input(shape=(input_shape[0], input_shape[1],), name="inputs")
    
    is_ready = False
    
    if masking:
        phi = Masking( mask_value=mask_v, name="masking_1")(inputs)
        is_ready=True
    
    if batchnorm: 
        if is_ready: 
            phi = BatchNormalization()(phi)
        else: 
            phi = BatchNormalization()(inputs)
            is_ready=True
    
    for iphi,phi_l in enumerate(phi_layers):
        if is_ready: 
            phi = TimeDistributed( Dense(phi_l,activation='relu'),name=f"Phi_{iphi}_Dense")(phi)
        else:
            phi = TimeDistributed( Dense(phi_l,activation='relu'),name=f"Phi_{iphi}_Dense")(inputs)
            is_ready=True
    
    if batchnorm: 
        phi = BatchNormalization()(phi)

    if add_selfattention:
        phi = Attention()([phi, phi], return_attention_scores=False)
        # query_value_attention_seq, attn_scores = Attention()([phi, phi], return_attention_scores=True)

    F = Sum()(phi)

    if batchnorm: 
        F = BatchNormalization()(F)
    
    for iF,F_l in enumerate(F_layers):
        F = Dense(F_l,activation='relu', name=f"F_{iF}_Dense")(F)

    if do_reg_out==0:
        F = Dense(1, activation='sigmoid', name='F_Output')(F)
    else:
        F = Dense(do_reg_out, activation='linear', name='F_Output')(F)
    
    model = Model(inputs=inputs, outputs=F)
    model.summary()

    if do_reg_out==0:
        my_loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(loss=my_loss, optimizer='adam', metrics=['accuracy'])
    else:
        my_loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
        model.compile(loss=my_loss, optimizer='adam', metrics=['mse'])

    return model

def model_deep_set_muonhit(input_shape, ll = 0.1,
        phi_layers=[50,50,50], 
        rho_layers=[50,50,50], 
        F_layers=[20,10], 
        batchnorm=True, mask_v=0, add_selfattention=False):
    """
    Implementation of deep sets model with muon+hit classifier.
    NOTE: A regression version of this network is not implemented yet.
    phi_layers correspond to the architecture of the hit-level 
    feature extraction network.
    F_layers correspond to the architecture of the event-level 
    classifier network.
    mask_v sets input values to be masked.
    add_selfattention adds a self-attention block.
    If do_reg_out==0, the network is trained as a binary classifier.
    If do_reg_out>0, the network is trained as a regressor with 
    do_reg_out outputs.
    
    """

    if len(phi_layers) < 1:
        print("Invalid phi_layers", phi_layers)
        return -1

    if len(rho_layers) < 1:
        print("Invalid rho_layers", rho_layers)
        return -1
    
    if len(F_layers) < 1:
        print("Invalid F_layers", F_layers)
        return -1
    
    inputs = Input(shape=(input_shape[0], input_shape[1],), name="inputs")
    
    phi = Masking( mask_value=mask_v, name="masking_1")(inputs)
    
    if batchnorm:
        phi = BatchNormalization()(phi)
    
    for iphi,phi_l in enumerate(phi_layers):
        phi = TimeDistributed( Dense(phi_l,activation='relu'),name=f"Phi_{iphi}_Dense")(phi)
    
    if batchnorm:
        phi = BatchNormalization()(phi)
    
    if add_selfattention:
        phi = Attention()([phi, phi], return_attention_scores=False)
        # query_value_attention_seq, attn_scores = Attention()([phi, phi], return_attention_scores=True)
    
    rho = TimeDistributed(Dense(rho_layers[0],activation='relu'),name=f"Rho_0_Dense")(phi)

    if len(rho_layers) > 1:
        for irho,rho_l in enumerate(rho_layers[1:]):
            rho = TimeDistributed( Dense(rho_l,activation='relu'),name=f"Phi_{irho+1}_Dense")(rho)

    rho = TimeDistributed(Dense(1,activation='sigmoid'),name=f"Rho_Output")(rho)
    all_rho = tf.keras.layers.Flatten()(rho)
        
    all_pool = tf.keras.layers.Concatenate(axis=1)([phi, rho])

    F = Sum()(all_pool)
    
    
    if batchnorm:
        F = BatchNormalization()(F)

    for iF,F_l in enumerate(F_layers):
        F = Dense(F_l,activation='relu', name=f"F_{iF}_Dense")(F)

    F = Dense(1, activation='sigmoid', name='F_Output')(F)
    
    total_output = tf.keras.layers.Concatenate(axis=1)([F, all_rho])
    
    model = Model(inputs=inputs, outputs=total_output)
    
    model.summary()
    
    my_loss = comp_loss(ll, input_shape[0]+1)
    model.compile(loss=my_loss, optimizer='adam')
    
    return model

def model_recurrent_muon(input_shape,
        rec_layer='lstm',
        rec_layers=[20], 
        F_layers=[20,10], 
        batchnorm=True, 
        mask_v=-99, 
        do_reg_out=0,
        masking=True):
    """
    Implementation of recurrent model with muon outputs only.
    rec_layer accepts 'lstm' or 'gru'.
    phi_layers correspond to the architecture of the hit-level 
    feature extraction network.
    F_layers correspond to the architecture of the event-level 
    classifier network.
    mask_v sets input values to be masked.
    add_selfattention adds a self-attention block.
    If do_reg_out==0, the network is trained as a binary classifier.
    If do_reg_out>0, the network is trained as a regressor with 
    do_reg_out outputs.
    
    """

    if len(rec_layers) < 1:
        print("Invalid rec_layers", rec_layers)
        return -1

    if 'lstm' not in rec_layer and 'gru' not in rec_layer:
        print('rec_layer must be either gru or lstm, it is',rec_layer)
        return -1
    
    if len(F_layers) < 1:
        print("Invalid F_layers", F_layers)
        return -1
    
    inputs = Input(shape=(input_shape[0], input_shape[1],), name="inputs")
    
    is_ready = False
    
    if masking:
        hidden = Masking( mask_value=mask_v, name="masking_1")(inputs)
        is_ready=True
    
    if batchnorm: 
        if is_ready: 
            hidden = BatchNormalization()(hidden)
        else: 
            hidden = BatchNormalization()(inputs)
            is_ready=True
    
    if 'gru' in rec_layer:
        for irec,rec_l in enumerate(rec_layers):
            if is_ready:
                hidden = GRU(rec_l, name=f'gru_{irec}')(hidden)
            else:
                hidden = GRU(rec_l, name=f'gru_{irec}')(inputs)
                is_ready=True
    if 'lstm' in rec_layer:
        for irec,rec_l in enumerate(rec_layers):
            if is_ready:
                hidden = LSTM(rec_l, name=f'lstm_{irec}')(hidden)
            else:
                hidden = LSTM(rec_l, name=f'lstm_{irec}')(inputs)
                is_ready=True
    
    for iF,F_l in enumerate(F_layers):
        hidden = Dense(F_l, activation='relu', name=f'F_dense_{iF}')(hidden)

    if do_reg_out==0:
        out = Dense(1, activation='sigmoid', name='output')(hidden)
    else:
        out = Dense(do_reg_out, activation='linear', name='output')(hidden)
    
    model = Model(inputs=inputs, outputs=out)
    model.summary()

    if do_reg_out==0:
        my_loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(loss=my_loss, optimizer='adam', metrics=['accuracy'])
    else:
        my_loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
        model.compile(loss=my_loss, optimizer='adam', metrics=['mse'])

    return model
    

def model_mlp_muon(input_shape,
        dense_layers=[100,50,10], 
        batchnorm=True, 
        do_reg_out=0):
    """
    Implementation of MLP model with muon outputs only.
    dense_layers correspond to the architecture of the ReLU MLP.
    If do_reg_out==0, the network is trained as a binary classifier.
    If do_reg_out>0, the network is trained as a regressor with 
    do_reg_out outputs.
    
    """

    if len(dense_layers) < 1:
        print("Invalid dense_layers", dense_layers)
        return -1
    
    inputs = Input(shape=(input_shape, ), name="inputs")
    
    if batchnorm: 
        hidden = BatchNormalization()(inputs)
    
    for iD,D_l in enumerate(dense_layers):
        if batchnorm==False and iD==0:
            hidden = Dense(D_l, activation='relu', name=f'F_dense_{iD}')(inputs)    
        else:
            hidden = Dense(D_l, activation='relu', name=f'F_dense_{iD}')(hidden)

    if do_reg_out==0:
        out = Dense(1, activation='sigmoid', name='output')(hidden)
    else:
        out = Dense(do_reg_out, activation='linear', name='output')(hidden)
    
    model = Model(inputs=inputs, outputs=out)
    model.summary()

    if do_reg_out==0:
        my_loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(loss=my_loss, optimizer='adam', metrics=['accuracy'])
    else:
        my_loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
        model.compile(loss=my_loss, optimizer='adam', metrics=['mse'])

    return model
    

def model_tcn_muon(input_shape,
        convs_1ds=[ (64,3,1), (64,3,1) ],
        F_layers=[20,10], 
        batchnorm_dense=False,
        batchnorm_conv=False, 
        do_reg_out=0,
        l1_reg=0, l2_reg=0,
        ll=0,
        pen_type=0, pen_x=True, pen_a=True, bkg_pen_x=False, bkg_pen_a=False,
        reg_bias=True,
        pooling='max',
        residual=False):

    inputs = Input(shape=(input_shape[0], input_shape[1],), name="inputs")
    
    def residual_block(irblock, x, nfilters, ksize, strides):
        ishape=(x.shape[1], x.shape[2])
        
        x = BatchNormalization()(x)
        
        fx = Conv1D(nfilters, kernel_size=ksize, strides=strides,
                kernel_initializer='variance_scaling', padding='same',
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                input_shape=ishape, name=f"RB_C1D_{irblock}"
               )(x)

        fx = BatchNormalization()(fx)
        
        fx = tf.keras.activations.relu(fx, max_value=1.0)
        
        fx = Conv1D(nfilters, kernel_size=ksize, strides=strides,
                    kernel_initializer='variance_scaling', padding='same',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                    name=f"RB_C1D_{irblock}_1"
                    )(fx)
        
        fx = BatchNormalization()(fx)
        
        if nfilters > ishape[1]:
            dim_diff = nfilters - ishape[1] - 1
            x = tf.pad( x, tf.constant([ [0, 0], [0, 0], [  dim_diff  , 1] ] ))
            
        out = Add()([x,fx])
        out = tf.keras.activations.relu(out, max_value=1.0)
        
        return out
    

    for ic1d,c1d in enumerate(convs_1ds):
        if residual:
            conv = residual_block(irblock=ic1d, x=inputs, nfilters=c1d[0], ksize=c1d[1], strides=c1d[2])
        else:
            if ic1d == 0:
                conv = Conv1D(filters=c1d[0], kernel_size=c1d[1], strides=c1d[2], 
                              activation='relu', kernel_initializer='variance_scaling', kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                              kernel_constraint=max_norm(1.),
                              input_shape=input_shape, name=f"C1D_{ic1d}" )(inputs)

                # if batchnorm_conv: conv = BatchNormalization()(conv)

            else:
                conv = Conv1D(filters=c1d[0], kernel_size=c1d[1], strides=c1d[2], 
                              activation='relu', kernel_initializer='variance_scaling', kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                              kernel_constraint=max_norm(1.),
                              name=f"C1D_{ic1d}" )(conv)

                # if batchnorm_conv: conv = BatchNormalization()(conv)

    if 'max' in pooling:
        hidden = GlobalMaxPooling1D()(conv)
    elif 'average' in pooling:
        hidden = GlobalAveragePooling1D()(conv)
    elif 'flat' in pooling:
        hidden = Flatten()(conv)

    for iF,F_l in enumerate(F_layers):
        hidden = Dense(F_l, activation='relu', kernel_initializer='variance_scaling', kernel_constraint=max_norm(1.), bias_constraint=max_norm(1.), name=f'F_dense_{iF}')(hidden)

    if batchnorm_dense: hidden = BatchNormalization()(hidden)
    
    if do_reg_out==0:
        out = Dense(1, activation='sigmoid', name='output')(hidden)
    else:
        outputs = []

        #out_clas = Dense(1, activation='sigmoid', name='output_clas')(hidden)
        #outputs.append(out_clas)

        #for ior in range(do_reg_out):
        #if do_reg_out > 0:
        outputs.append(Dense(do_reg_out+1, activation='linear', kernel_initializer='variance_scaling', 
                             kernel_constraint=max_norm(1.), bias_constraint=max_norm(1.),
                                 use_bias=reg_bias, name=f'output')(hidden))

        # out = concatenate(outputs, name="combined_output")

    if do_reg_out==0:
        model = Model(inputs=inputs, outputs=out)
        model.summary()
    
        my_loss = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size')
        opt = Adam(learning_rate=0.01)
        model.compile(loss=my_loss, optimizer=opt, metrics=['accuracy'])
    else:
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        #closs = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size') 
        #rloss = regr_loss(do_angle=(do_reg_out>1), pen_type=0, pen_x=True, pen_a=True, bkg_pen_x=False, bkg_pen_a=False)
        combloss = class_and_regr_loss(ll, do_angle=(do_reg_out>1), pen_type=pen_type, pen_x=pen_x, pen_a=pen_a, bkg_pen_x=bkg_pen_x, bkg_pen_a=bkg_pen_a, linearized=True)

        opt = Adam(learning_rate=0.001)
        # model.compile(loss=[closs, rloss], loss_weights=[1,1], optimizer=opt)
        model.compile(loss=combloss, optimizer=opt)

    return model
