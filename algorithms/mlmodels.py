from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer, Masking, Input, Dense, Concatenate, Add
from tensorflow.keras.layers import ReLU, BatchNormalization, Attention, ReLU
from tensorflow.keras.layers import BatchNormalization, Embedding, Lambda, TimeDistributed
from tensorflow.keras.layers import LSTM, GRU, Conv1D, GlobalMaxPooling1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm

import tensorflow as tf

import sys

def regr_loss_xa_quant(use_pen=True, pen_type=0):
    """
    Custom loss that adds an event-based binary classifier loss (muon vs no muon)
    and many hit-based binary classifier losses (muon hit vs noise hit).
    The two components are importance weighted by input ll.

    """
    
    half = tf.constant(0.5, dtype=tf.float32)
    max_val = tf.constant(1, dtype=tf.float32)
    min_val = tf.constant(-1, dtype=tf.float32)
    pow_val = tf.constant(2, dtype=tf.float32)
    delta_x = tf.constant(0.01, dtype=tf.float32)
    delta_a = tf.constant(0.01, dtype=tf.float32)
    q_quant = tf.constant(0.25, dtype=tf.float32)
    
    reg_loss = tf.keras.losses.Huber(reduction='sum_over_batch_size')
    
    def quant_loss(ytrue, ypred, q):
        e = ytrue - ypred
        return tf.math.reduce_mean(K.maximum(q*e, (q-1)*e), axis=-1)
    
    def penf(pred, bound_val):
        if pen_type == 2:
            pfunc = tf.math.abs( pred - bound_val ) + tf.math.pow( pred - bound_val, pow_val )
        elif pen_type == 1:
            pfunc = tf.math.abs( pred - bound_val )
        else:
            pfunc = tf.math.pow( pred - bound_val, pow_val )
        
        if bound_val<0:
            pfunc = pfunc*( tf.cast( tf.math.less( pred, bound_val ), dtype=tf.float32 ) )
        else:
            pfunc = pfunc*( tf.cast( tf.math.greater( pred, bound_val ), dtype=tf.float32 ) )
            
        return tf.math.reduce_mean(pfunc, axis=-1)
    
    def tot_loss_indx(y_true, y_pred_nom, y_pred_quant, delta=tf.constant(1., dtype=tf.float32)):
        r_loss = reg_loss( y_true, y_pred_nom, delta )
        q_loss = quant_loss( y_true, y_pred_quant, q_quant)
        p_loss_1 = penf(y_pred_nom, max_val)
        p_loss_2 = penf(y_pred_nom, min_val)
        
        return tf.math.add_n( [r_loss, q_loss, p_loss_1, p_loss_2] )

    def model_loss(y_true, y_pred):
        
        x_loss = tot_loss_indx( y_true[:,0], y_pred[:,0], y_pred[:,1], delta=delta_x )
        a_loss = tot_loss_indx( y_true[:,1], y_pred[:,2], y_pred[:,3], delta=delta_a )

        return tf.math.add(x_loss, a_loss)

    return model_loss

def regr_loss(n_outs=3, use_pen=True, pen_type=0, weights=[1,1,0.1]):
    """
    Custom loss that adds an event-based binary classifier loss (muon vs no muon)
    and many hit-based binary classifier losses (muon hit vs noise hit).
    The two components are importance weighted by input ll.

    """
    
    print('*#&!(@*&# n_outs', n_outs)
    
    half = tf.constant(0.5, dtype=tf.float32)
    max_val = tf.constant(1, dtype=tf.float32)
    min_val = tf.constant(-1, dtype=tf.float32)
    pow_val = tf.constant(2, dtype=tf.float32)
    reduct = tf.constant(1./100., dtype=tf.float32)
    cweights = [ tf.constant(ww, dtype=tf.float32) for ww in weights ]
    
    reg_loss = tf.keras.losses.MeanAbsolutePercentageError(reduction='sum_over_batch_size')
    
    def penf(pred, bound_val):
        if pen_type == 2:
            pfunc = tf.math.abs( pred - bound_val ) + tf.math.pow( pred - bound_val, pow_val )
        elif pen_type == 1:
            pfunc = tf.math.abs( pred - bound_val )
        else:
            pfunc = tf.math.pow( pred - bound_val, pow_val )
        
        if bound_val<0:
            return pfunc*( tf.cast( tf.math.less( pred, bound_val ), dtype=tf.float32 ) )
        else:
            return pfunc*( tf.cast( tf.math.greater( pred, bound_val ), dtype=tf.float32 ) )
        
    def fadd(t1, t2, w=tf.constant(1, dtype=tf.float32) ):
        def ADD():
            return w*tf.math.add( t1, t2 )
        return ADD
        
    def fret(this_tensor):
        def RET():
            return this_tensor
        return RET

    def model_loss(y_true, y_pred):
        
        # regression losses
        regrloss_val = cweights[0]*reg_loss ( y_true[:,0], y_pred[:,0] )*reduct
        regrloss_val = tf.cond( tf.constant(n_outs>1, dtype=tf.bool), 
                               true_fn=fadd(regrloss_val, reg_loss ( y_true[:,1], y_pred[:,1] ), cweights[1]), 
                               false_fn=fret(regrloss_val)  )*reduct
        regrloss_val = tf.cond( tf.constant(n_outs>2, dtype=tf.bool), 
                               true_fn=fadd(regrloss_val, reg_loss ( y_true[:,2], y_pred[:,2] ), cweights[2]),
                               false_fn=fret(regrloss_val)  )*reduct
        
        # penalties
        regrloss_val = tf.add( regrloss_val, cweights[0]*tf.add( penf(y_pred[:,0], max_val), penf(y_pred[:,0], min_val) ) )
        regrloss_val = tf.cond( tf.constant(n_outs>1, dtype=tf.bool), 
                               true_fn=fadd( regrloss_val, cweights[1]*tf.add( penf(y_pred[:,1], max_val), penf(y_pred[:,1], min_val) ) ),
                               false_fn=fret(regrloss_val)  )
        regrloss_val = tf.cond( tf.constant(n_outs>2, dtype=tf.bool), 
                               true_fn=fadd( regrloss_val, cweights[2]*tf.add( penf(y_pred[:,2], max_val), penf(y_pred[:,2], min_val) ) ),
                               false_fn=fret(regrloss_val)  )

        return regrloss_val

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
    

class MyTCNModel:
    def __init__(self, input_shape,
        convs_1ds=[ (64,3,1), (64,3,1) ],
        F_layers=[20,10], 
        batchnorm_dense=False,
        batchnorm_conv=False, 
        batchnorm_inputs=False,
        batchnorm_constr=False,
        n_outs=3,
        l1_reg=0, l2_reg=0,
        use_pen=True, pen_type=0,
        do_bias=True,
        pooling='max',
        learning_rate=0.001):
        
        self.input_shape=input_shape
        self.convs_1ds=convs_1ds
        self.F_layers=F_layers
        self.batchnorm_dense=batchnorm_dense
        self.batchnorm_conv=batchnorm_conv
        self.batchnorm_inputs=batchnorm_inputs
        self.batchnorm_constr=batchnorm_constr
        self.n_outs=n_outs
        self.l1_reg=l1_reg
        self.l2_reg=l2_reg
        self.use_pen=use_pen
        self.pen_type=pen_type
        self.do_bias=do_bias
        self.pooling=pooling
        self.learning_rate = learning_rate
    
    def MyLoss(self):
        # return regr_loss(n_outs=self.n_outs, use_pen=self.use_pen, pen_type=self.pen_type)
        return regr_loss_xa_quant(use_pen=self.use_pen, pen_type=self.pen_type)

    def model(self):

        inputs = Input(shape=(self.input_shape[0], self.input_shape[1],), name="inputs")
        
    
        def MyBatchNorm():
            if self.batchnorm_constr:
                return BatchNormalization(beta_constraint=max_norm(1.), gamma_constraint=max_norm(1.))
            else:
                return BatchNormalization()
    
        def residual_block(irblock, x, nfilters, ksize, strides):
            ishape=(x.shape[1], x.shape[2])

            x = MyBatchNorm()(x)

            fx = Conv1D(nfilters, kernel_size=ksize, strides=strides,
                    kernel_initializer='variance_scaling', padding='same',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg),
                    kernel_constraint=max_norm(1.),
                    input_shape=ishape, name=f"RB_C1D_{irblock}"
                   )(x)

            fx = MyBatchNorm()(fx)

            fx = ReLU()(fx)

            fx = Conv1D(nfilters, kernel_size=ksize, strides=strides,
                        kernel_initializer='variance_scaling', padding='same',
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                        kernel_constraint=max_norm(1.),
                        name=f"RB_C1D_{irblock}_1"
                        )(fx)

            fx = MyBatchNorm()(fx)

            if nfilters > ishape[1]:
                dim_diff = nfilters - ishape[1] - 1
                x = tf.pad( x, tf.constant([ [0, 0], [0, 0], [  dim_diff  , 1] ] ))
            elif nfilters < ishape[1]:
                dim_diff = ishape[1] - nfilters - 1
                fx = tf.pad( fx, tf.constant([ [0, 0], [0, 0], [  dim_diff  , 1] ] ))

            out = Add()([x,fx])
            fx = ReLU()(out)

            return out
    
        if self.batchnorm_inputs: 
            conv = MyBatchNorm()(inputs)
        else:
            conv = inputs      

        for ic1d,c1d in enumerate(self.convs_1ds):
            if c1d[3] == 0:
                conv = Conv1D(filters=c1d[0], kernel_size=c1d[1], strides=c1d[2], 
                                  activation='relu', kernel_initializer='variance_scaling', kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg),
                                  kernel_constraint=max_norm(1.),
                                  input_shape=(conv.shape[1], conv.shape[2]), name=f"C1D_{ic1d}" )(conv)
            if c1d[3] == 1:
                conv = residual_block(irblock=ic1d, x=inputs, nfilters=c1d[0], ksize=c1d[1], strides=c1d[2])

            if self.batchnorm_conv: conv = MyBatchNorm()(conv)


        if 'max' in self.pooling:
            hidden = GlobalMaxPooling1D()(conv)
        elif 'average' in self.pooling:
            hidden = GlobalAveragePooling1D()(conv)
        elif 'flat' in self.pooling:
            hidden = Flatten()(conv)

        for iF,F_l in enumerate(self.F_layers):
            hidden = Dense(F_l, activation='relu', kernel_initializer='variance_scaling', kernel_constraint=max_norm(1.), bias_constraint=max_norm(1.), name=f'F_dense_{iF}')(hidden)

        if self.batchnorm_dense: hidden = MyBatchNorm()(hidden)
    
        out = Dense(self.n_outs, activation='linear', kernel_initializer='variance_scaling', 
                                 kernel_constraint=max_norm(1.), bias_constraint=max_norm(1.),
                                     use_bias=self.do_bias, name=f'output')(hidden)

        model = Model(inputs=inputs, outputs=[out])
        model.summary()

        custom_loss = self.MyLoss()

        opt = Adam(learning_rate=self.learning_rate)
        model.compile(loss=custom_loss, optimizer=opt)

        return model
