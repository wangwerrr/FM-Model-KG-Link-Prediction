import tensorflow as tf
from tensorflow.python.framework import ops
from . import utils
import math
from scipy.stats import rankdata
import numpy as np
from .utils import sigmoid

n_entity = 14951
n_relation = 1345

class TFFMCore():
    """
    This class implements underlying routines about creating computational graph.

    Its required `n_features` to be set at graph building time.


    Parameters
    ----------
    order : int, default: 2
        Order of corresponding polynomial model.
        All interaction from bias and linear to order will be included.

    rank : int, default: 5
        Number of factors in low-rank appoximation.
        This value is shared across different orders of interaction.

    input_type : str, 'dense' or 'sparse', default: 'dense'
        Type of input data. Only numpy.array allowed for 'dense' and
        scipy.sparse.csr_matrix for 'sparse'. This affects construction of
        computational graph and cannot be changed during training/testing.

    loss_function : function: (tf.Op, tf.Op) -> tf.Op, default: None
        Loss function.
        Take 2 tf.Ops: outputs and targets and should return tf.Op of loss
        See examples: .utils.loss_mse, .utils.loss_logistic

    optimizer : tf.train.Optimizer, default: AdamOptimizer(learning_rate=0.01)
        Optimization method used for training

    reg : float, default: 0
        Strength of L2 regularization

    use_diag : bool, default: False
        Use diagonal elements of weights matrix or not.
        In the other words, should terms like x^2 be included.
        Ofter reffered as a "Polynomial Network".
        Default value (False) corresponds to FM.

    reweight_reg : bool, default: False
        Use frequency of features as weights for regularization or not.
        Should be usefull for very sparse data and/or small batches

    init_std : float, default: 0.01
        Amplitude of random initialization

    seed : int or None, default: None
        Random seed used at graph creating time


    Attributes
    ----------
    graph : tf.Graph or None
        Initialized computational graph or None

    trainer : tf.Op
        TensorFlow operation node to perform learning on single batch

    n_features : int
        Number of features used in this dataset.
        Inferred during the first call of fit() method.

    saver : tf.Op
        tf.train.Saver instance, connected to graph

    summary_op : tf.Op
        tf.merge_all_summaries instance for export logging

    b : tf.Variable, shape: [1]
        Bias term.

    w : array of tf.Variable, shape: [order]
        Array of underlying representations.
        First element will have shape [n_features, 1],
        all the others -- [n_features, rank].

    Notes
    -----
    Parameter `rank` is shared across all orders of interactions (except bias and
    linear parts).
    tf.sparse_reorder doesn't requied since COO format is lexigraphical ordered.
    This implementation uses a generalized approach from referenced paper along
    with caching.

    References
    ----------
    Steffen Rendle, Factorization Machines
        http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """
    def __init__(self, e_rank=100, r_rank=100, input_type='sparse', loss_function=utils.loss_cross_entropy,
                optimizer=tf.train.AdamOptimizer(learning_rate=0.01), reg=0, init_std=0.1,
                use_diag=False, reweight_reg=False, seed=None):
        self.e_rank = e_rank
        self.r_rank = r_rank
        self.use_diag = use_diag
        self.input_type = input_type
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.reg = reg
        self.reweight_reg = reweight_reg
        self.init_std = init_std
        self.seed = seed
        self.n_features = None
        self.graph = None

        self.ranking = None

    def set_num_features(self, n_features):
        self.n_features = n_features

    # initializing variables
    def init_learnable_params(self):
        rnd_weights = tf.random_normal([n_entity, self.e_rank], mean=0.0, stddev=self.init_std)
        self.Ve = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, trainable=True, name='embedding_Ve'), msg='NaN or Inf in Ve')
        rnd_weights = tf.random_normal([n_relation, self.r_rank], mean=0.0, stddev=self.init_std)
        self.Vr = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, trainable=True, name='embedding_Vr'), msg='NaN or Inf in Vr')

        rnd_weights = tf.random_normal([self.e_rank], mean=0.0, stddev=self.init_std)
        self.Wsr = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, trainable=True, name='Wsr'), msg='NaN or Inf in Vr')
        rnd_weights = tf.random_normal([self.e_rank], mean=0.0, stddev=self.init_std)
        self.Wro = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, trainable=True, name='Wro'), msg='NaN or Inf in Vr')
        rnd_weights = tf.random_normal([self.e_rank], mean=0.0, stddev=self.init_std)
        self.Wso = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, trainable=True, name='Wso'), msg='NaN or Inf in Vr')

        rnd_weights = tf.random_normal([self.e_rank], mean=0.0, stddev=self.init_std)
        self.Wss = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, trainable=True, name='Wss'), msg='NaN or Inf in Vr')
        rnd_weights = tf.random_normal([self.e_rank], mean=0.0, stddev=self.init_std)
        self.Woo = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, trainable=True, name='Woo'), msg='NaN or Inf in Vr')
        rnd_weights = tf.random_normal([self.e_rank], mean=0.0, stddev=self.init_std)
        self.Wrr = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, trainable=True, name='Wrr'), msg='NaN or Inf in Vr')

        rnd_weights = tf.random_normal([self.r_rank, self.e_rank], mean=0.0, stddev=self.init_std)
        self.T = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights, trainable=True, name='T'), msg='NaN or Inf in T')

        tf.add_to_collection('Ve', self.Ve)
        tf.add_to_collection('Vr', self.Vr)
        tf.add_to_collection('Wsr', self.Wsr)
        tf.add_to_collection('Wro', self.Wro)
        tf.add_to_collection('Wso', self.Wso)
        tf.add_to_collection('Wss', self.Wss)
        tf.add_to_collection('Wrr', self.Wrr)
        tf.add_to_collection('Woo', self.Woo)
        tf.add_to_collection('T', self.T)


    # initializing placeholders for input data
    def init_placeholders(self):
        if self.input_type == 'dense':
            self.train_x = tf.placeholder(tf.float32, shape=[None, self.n_features], name='x')
            tf.add_to_collection('x', self.train_x)

        else:
            with tf.name_scope('sparse_placeholders') as scope:
                self.xs_raw_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
                self.xs_raw_values = tf.placeholder(tf.float32, shape=[None], name='raw_values')
                self.xs_raw_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
            # tf.sparse_reorder is not needed since scipy return COO in canonical order
            self.train_xs = tf.SparseTensor(self.xs_raw_indices, self.xs_raw_values, self.xs_raw_shape)
            tf.add_to_collection('xs_raw_indices', self.xs_raw_indices)
            tf.add_to_collection('xs_raw_values', self.xs_raw_values)
            tf.add_to_collection('xs_raw_shape', self.xs_raw_shape)

            with tf.name_scope('sparse_placeholders') as scope:
                self.xr_raw_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
                self.xr_raw_values = tf.placeholder(tf.float32, shape=[None], name='raw_values')
                self.xr_raw_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
            # tf.sparse_reorder is not needed since scipy return COO in canonical order
            self.train_xr = tf.SparseTensor(self.xr_raw_indices, self.xr_raw_values, self.xr_raw_shape)
            tf.add_to_collection('xr_raw_indices', self.xr_raw_indices)
            tf.add_to_collection('xr_raw_values', self.xr_raw_values)
            tf.add_to_collection('xr_raw_shape', self.xr_raw_shape)

            with tf.name_scope('sparse_placeholders') as scope:
                self.xo_raw_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
                self.xo_raw_values = tf.placeholder(tf.float32, shape=[None], name='raw_values')
                self.xo_raw_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
            # tf.sparse_reorder is not needed since scipy return COO in canonical order
            self.train_xo = tf.SparseTensor(self.xo_raw_indices, self.xo_raw_values, self.xo_raw_shape)
            tf.add_to_collection('xo_raw_indices', self.xo_raw_indices)
            tf.add_to_collection('xo_raw_values', self.xo_raw_values)
            tf.add_to_collection('xo_raw_shape', self.xo_raw_shape)

        self.train_y = tf.placeholder(tf.float32, shape=[None], name='Y')
        tf.add_to_collection('Y', self.train_y)

    # calculating score
    def init_main_block(self):

        self.outputs = 0.0

        self.train_xsr = tf.sparse_concat(axis=1, sp_inputs=[self.train_xs, self.train_xr])
        self.train_xro = tf.sparse_concat(axis=1, sp_inputs=[self.train_xr, self.train_xo])
        self.train_xso = tf.sparse_concat(axis=1, sp_inputs=[self.train_xs, self.train_xo])

        Vr = utils.matmul_wrapper(self.Vr, self.T, 'dense')

        V = tf.concat(values=[self.Ve, Vr], axis=0)
        raw_dot = utils.matmul_wrapper(self.train_xsr, V, self.input_type)
        sum_pow = tf.pow(raw_dot, 2)
        x_pow = utils.pow_wrapper(self.train_xsr, 2, self.input_type)
        V_pow = tf.pow(V, 2)
        pow_sum = utils.matmul_wrapper(x_pow, V_pow, self.input_type)
        SR = (sum_pow - pow_sum) * self.Wsr / float(2.0)

        V = tf.concat(values=[Vr, self.Ve], axis=0)
        raw_dot = utils.matmul_wrapper(self.train_xro, V, self.input_type)
        sum_pow = tf.pow(raw_dot, 2)
        x_pow = utils.pow_wrapper(self.train_xro, 2, self.input_type)
        V_pow = tf.pow(V, 2)
        pow_sum = utils.matmul_wrapper(x_pow, V_pow, self.input_type)
        RO = (sum_pow - pow_sum) * self.Wro / float(2.0)

        V = tf.concat(values=[self.Ve, self.Ve], axis=0)
        raw_dot = utils.matmul_wrapper(self.train_xso, V, self.input_type)
        sum_pow = tf.pow(raw_dot, 2)
        x_pow = utils.pow_wrapper(self.train_xso, 2, self.input_type)
        V_pow = tf.pow(V, 2)
        pow_sum = utils.matmul_wrapper(x_pow, V_pow, self.input_type)
        SO = (sum_pow - pow_sum) * self.Wso / float(2.0)

        SS = tf.pow(utils.matmul_wrapper(self.train_xs, self.Ve, self.input_type), 2) * self.Wss
        OO = tf.pow(utils.matmul_wrapper(self.train_xo, self.Ve, self.input_type), 2) * self.Woo
        RR = tf.pow(utils.matmul_wrapper(self.train_xr, Vr, self.input_type), 2) * self.Wrr
        sum = SR + RO + SO + SS + OO + RR

        self.outputs = tf.reshape(tf.reduce_sum(sum, [1]), [-1, 1])

        tf.add_to_collection('outputs', self.outputs)

    #reg
    def init_regularization(self):
        with tf.name_scope('regularization') as scope:
            self.regularization = tf.reduce_sum(tf.pow(self.Ve, 2))
            self.regularization += tf.reduce_sum(tf.pow(self.Vr, 2))
            self.regularization += tf.reduce_sum(tf.pow(self.Wsr, 2))
            self.regularization += tf.reduce_sum(tf.pow(self.Wro, 2))
            self.regularization += tf.reduce_sum(tf.pow(self.Wso, 2))
            self.regularization += tf.reduce_sum(tf.pow(self.Wss, 2))
            self.regularization += tf.reduce_sum(tf.pow(self.Woo, 2))
            self.regularization += tf.reduce_sum(tf.pow(self.Wrr, 2))
            self.regularization += tf.reduce_sum(tf.pow(self.T, 2))
            self.regularization *= (1. / tf.cast(tf.pow(n_relation+n_entity, 2), tf.float32))

            tf.summary.scalar('regularization_penalty', self.regularization)

    # loss
    def init_loss(self):
        with tf.name_scope('loss') as scope:
            self.loss = self.loss_function(self.train_y, self.outputs)
            self.reduced_loss = tf.reduce_mean(self.loss)
            tf.summary.scalar('loss', self.reduced_loss)

    # target = loss+regularization
    def init_target(self):
        with tf.name_scope('target') as scope:
            self.target = self.reduced_loss + self.reg * self.regularization
            self.checked_target = tf.verify_tensor_all_finite(
                self.target,
                msg='NaN or Inf in target value',
                name='target')
            tf.summary.scalar('target', self.checked_target)

    def init_ranking(self):
        with tf.name_scope('ranking') as scope:
            self.n_entity = tf.constant(n_entity, dtype=tf.int64, name='n_entity')
            self.ranking = tf.py_func(rankdata, inp=[self.outputs, 'max'], Tout=tf.int64)
            self.ranking = self.n_entity + 1 - self.ranking
            tf.add_to_collection('n_entity', self.n_entity)
           # tf.summary.scalar(['valid_rank'], self.valid_rk)

    # building the graph
    def build_graph(self):
        """Build computational graph according to params."""
        assert self.n_features is not None, 'Number of features is unknown. It can be set explicitly by .core.set_num_features'
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            with tf.name_scope('learnable_params') as scope:
                self.init_learnable_params()
            with tf.name_scope('input_block') as scope:
                self.init_placeholders()
            with tf.name_scope('main_block') as scope:
                self.init_main_block()
            with tf.name_scope('optimization_criterion') as scope:
                self.init_regularization()
                self.init_loss()
                self.init_target()
                self.init_ranking()
            self.trainer = self.optimizer.minimize(self.checked_target)
            self.init_all_vars = tf.global_variables_initializer()
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()

            tf.add_to_collection('target', self.checked_target)
            tf.add_to_collection('trainer', self.trainer)
            tf.add_to_collection('summary_op', self.summary_op)

