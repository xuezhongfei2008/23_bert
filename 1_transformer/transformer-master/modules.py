# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def Mask(inputs, seq_len, mode='mul'):
    '''
    inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
    seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
    mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
    add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
    '''
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        print("mask1",mask)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        print("mask2",mask)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
            
    return outputs
    

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs



def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        # print("queries",len(queries.shape)-2)
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)  32,10,512   #units：输出的维度大小，改变inputs的最后一维
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        '''首先对queries，keys以及values进行全连接的变换，变换后的shape分别为(N, T_q, C)，(N, T_k, C)以及(N, T_k, C)'''
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        # print("Q_",Q_)
        '''将之前变换后的Q，K，V分为num_heads份，并将这些分开的张量重新在第一个维度拼接起来进行后续的运算。
        形成了新的Q_,K_,V_,其shape为h*N, T_q, C/h)，(h*N, T_k,C/h)以及(h*N, T_k,C/h).'''

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        # print("outputs1",outputs)
        '''将张量K_transopose之后和Q_进行了矩阵乘法的操作，其实就是attention计算时算attention score的一个方法，即向量的点乘。
        这里是把所有向量一起操作。,单词与单词的相似性'''
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        print("outputs2",outputs)
        '''然后再对该输出进行论文中所提到的scale操作，outputs的shape为[h*N, T_q, T_k].'''

        # print("outputs",outputs[])

        # outputs=Mask(outputs,outputs[:,0,0],mode='add')

        # Key Masking
        '''想让那些key值的unit为0的key对应的attention score极小，这样在加权计算value的时候相当于对结果不造成影响'''
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        '''
                reduce_sum(keys, axis=-1))将最后一个维度上的值加起来，keys的shape也从[N, T_k, C_k]变为[N,T_k]
                abs取绝对值，即其值只能为0（一开始的keys值第三个维度值全部为0，reduce_sum加起来之后为0），
                                    或正数（一开始的keys值第三个维度值并非全为0，reduce_sum加起来之后为非零数取绝对值为正数）
                tf.sign(x, name=None)，该函数返回符号 y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0，
                                                    sign会将原tensor对应的每个值变为-1,0,或者1。
            则经此操作，得到key_masks,有两个值，0或者1。0代表原先的keys第三维度所有值都为0，反之则为1，我们要mask的就是这些为0的key。
          '''
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        '''
            tf.tile(key_masks, [num_heads, 1])就把原来的shape为(N, T_k)的key_masks转化为shape为(h*N, T_k)的key_masks。
            （扩充第一个维度的作用是要与之前的split操作及concat操作保持一直，也就是对应多头的attention）。
        '''
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        '''tf.tile操作，由于每个queries都要对应这些keys，而mask的key对每个queries都是mask的。
            而之前的key_masks只相当于一份mask，所以扩充之前key_masks的维度，在中间加上一个维度大小为queries的序列长度。
            然后利用tile函数复制相同的mask值即可
        '''
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
        '''定义一个和outputs同shape的paddings，该tensor每个值都设定的极小。
            用where函数比较，当对应位置的key_masks值为0也就是需要mask时，outputs的该值（attention score）设置为极小的值（利用paddings实现），否则保留原来的outputs值。
            经过以上key mask操作之后outputs的shape仍为 (h*N, T_q, T_k)，只是对应mask了的key的score变为很小的值
        '''
        print("outputs3",outputs)
        # Causality = Future blinding
        if causality:
            '''causality参数告知我们是否屏蔽未来序列的信息（解码器self attention的时候不能看到自己之后的那些信息）'''
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            # tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
            '''一个和outputs后两维的shape相同shape（T_q,T_k）的一个张量（矩阵）。
            然后将该矩阵转为三角阵tril。三角阵中，对于每一个T_q,凡是那些大于它角标的T_k值全都为0，
            这样作为mask就可以让query只取它之前的key（self attention中query即key）。
            由于该规律适用于所有query，接下来仍用tile扩展堆叠其第一个维度，构成masks，shape为(h*N, T_q,T_k).
            
            之后两行代码进行paddings，和之前key mask的过程一样就不多说了。
            以上操作就可以当不需要来自未来的key值时将未来位置的key的score设置为极小
            '''
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
        '''将attention score了利用softmax转化为加起来为1的权值'''
        # Query Masking
        '''本身不携带信息或者暂时禁止利用其信息的内容'''
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        '''query mask也是要将那些初始值为0的queryies（比如一开始句子被PAD填充的那些位置作为query） mask住'''
        outputs *= query_masks # broadcasting. (h*N, T_q, T_k)
        '''outputs的值和query_masks相乘。这里的outputs是之前已经softmax之后的权值。所以此步之后，
        需要mask的权值会乘以0，不需要mask的乘以之前取的正数的sign为1所以权值不变
        扩展维度等是在最后一个维度展开的。操作之后形成的query_masks的shape为[h*N, T_q, T_k]'''
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
        '''之后是residual操作加上inputs残差，'''
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
        '''然后是normalize'''
 
    return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = normalize(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)
    
    

            
