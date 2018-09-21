from __future__ import absolute_import
import tensorflow as tf
from .core import TFFMCore
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
import six
from tqdm import tqdm
import numpy as np
import os
import sklearn
from scipy.sparse import csr_matrix, coo_matrix, hstack
import timeit
from scipy.stats import rankdata
from .utils import load_obj

n_entity = 14951
n_relation = 1345


def batcher(Xs_, Xr_, Xo_, y_=None, batch_size=-1):
    """Split data to mini-batches.

    Parameters
    ----------
    X_ : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y_ : np.array or None, shape (n_samples,)
        Target vector relative to X.

    batch_size : int
        Size of batches.
        Use -1 for full-size batches

    Yields
    -------
    ret_x : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Same type as input
    ret_y : np.array or None, shape (batch_size,)
    """
    n_samples = Xs_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_xs = Xs_[i:upper_bound]
        ret_xr = Xr_[i:upper_bound]
        ret_xo = Xo_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
        yield (ret_xs, ret_xr, ret_xo, ret_y)



def batch_to_feeddict(Xs, Xr, Xo, y, core):
    """Prepare feed dict for session.run() from mini-batch.
    Convert sparse format into tuple (indices, values, shape) for tf.SparseTensor
    Parameters
    ----------
    X : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Training vector, where batch_size in the number of samples and
        n_features is the number of features.
    y : np.array, shape (batch_size,)
        Target vector relative to X.
    core : TFFMCore
        Core used for extract appropriate placeholders
    Returns
    -------
    fd : dict
        Dict with formatted placeholders
    """
    fd = {}
    if core.input_type == 'dense':
        fd[core.train_x] = Xs.astype(np.float32)
    else:
        # sparse case
        Xs_sparse = Xs.tocoo()
        fd[core.xs_raw_indices] = np.hstack(
            (Xs_sparse.row[:, np.newaxis], Xs_sparse.col[:, np.newaxis])
        ).astype(np.int64)
        fd[core.xs_raw_values] = Xs_sparse.data.astype(np.float32)
        fd[core.xs_raw_shape] = np.array(Xs_sparse.shape).astype(np.int64)

        Xr_sparse = Xr.tocoo()
        fd[core.xr_raw_indices] = np.hstack(
            (Xr_sparse.row[:, np.newaxis], Xr_sparse.col[:, np.newaxis])
        ).astype(np.int64)
        fd[core.xr_raw_values] = Xr_sparse.data.astype(np.float32)
        fd[core.xr_raw_shape] = np.array(Xr_sparse.shape).astype(np.int64)

        Xo_sparse = Xo.tocoo()
        fd[core.xo_raw_indices] = np.hstack(
            (Xo_sparse.row[:, np.newaxis], Xo_sparse.col[:, np.newaxis])
        ).astype(np.int64)
        fd[core.xo_raw_values] = Xo_sparse.data.astype(np.float32)
        fd[core.xo_raw_shape] = np.array(Xo_sparse.shape).astype(np.int64)
    if y is not None:
        fd[core.train_y] = y.astype(np.float32)
    return fd


class TFFMBaseModel(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for FM.
    This class implements L2-regularized arbitrary order FM model.

    It supports arbitrary order of interactions and has linear complexity in the
    number of features (a generalization of the approach described in Lemma 3.1
    in the referenced paper, details will be added soon).

    It can handle both dense and sparse input. Only numpy.array and CSR matrix are
    allowed as inputs; any other input format should be explicitly converted.

    Support logging/visualization with TensorBoard.


    Parameters (for initialization)
    ----------
    batch_size : int, default: -1
        Number of samples in mini-batches. Shuffled every epoch.
        Use -1 for full gradient (whole training set in each batch).

    n_epoch : int, default: 100
        Default number of epoches.
        It can be overrived by explicitly provided value in fit() method.

    log_dir : str or None, default: None
        Path for storing model stats during training. Used only if is not None.
        WARNING: If such directory already exists, it will be removed!
        You can use TensorBoard to visualize the stats:
        `tensorboard --logdir={log_dir}`

    session_config : tf.ConfigProto or None, default: None
        Additional setting passed to tf.Session object.
        Useful for CPU/GPU switching, setting number of threads and so on,
        `tf.ConfigProto(device_count = {'GPU': 0})` will disable GPU (if enabled)

    verbose : int, default: 0
        Level of verbosity.
        Set 1 for tensorboard info only and 2 for additional stats every epoch.

    kwargs : dict, default: {}
        Arguments for TFFMCore constructor.
        See TFFMCore's doc for details.

    Attributes
    ----------
    core : TFFMCore or None
        Computational graph with internal utils.
        Will be initialized during first call .fit()

    session : tf.Session or None
        Current execution session or None.
        Should be explicitly terminated via calling destroy() method.

    steps : int
        Counter of passed learning epochs, used as step number for writing stats

    n_features : int
        Number of features used in this dataset.
        Inferred during the first call of fit() method.

    intercept : float, shape: [1]
        Intercept (bias) term.

    weights : array of np.array, shape: [order]
        Array of underlying representations.
        First element will have shape [n_features, 1],
        all the others -- [n_features, rank].

    Notes
    -----
    You should explicitly call destroy() method to release resources.
    See TFFMCore's doc for details.
    """


    def init_basemodel(self, epochs=50, batch_size=-1, negative_sample=1, log_dir=None, session_config=None, verbose=1, seed=None, **core_arguments):
        core_arguments['seed'] = seed
        self.core = TFFMCore(**core_arguments)
        self.batch_size = batch_size
        self.epochs = epochs
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        self.session_config = tf.ConfigProto(allow_soft_placement=True)
        self.session_config.gpu_options.allow_growth = True
        self.verbose = verbose
        self.steps = 0
        self.seed = seed
        self.negative_sample = negative_sample

    def initialize_session(self):
        """Start computational session on built graph.
        Initialize summary logger (if needed).
        """
        # if self.core.graph is None:
        #    raise 'Graph not found. Try call .core.build_graph() before .initialize_session()'
        if self.need_logs:
            self.summary_writer = tf.summary.FileWriter(self.log_dir, self.core.graph)
            if self.verbose > 0:
                full_log_path = os.path.abspath(self.log_dir)
                print('Initialize logs, use: \ntensorboard --logdir={}'.format(full_log_path))

        self.session = tf.Session(config=self.session_config, graph=self.core.graph)
        self.session.run(self.core.init_all_vars)

    @abstractmethod
    def preprocess_target(self, target):
        """Prepare target values to use."""

    # training
    def fit(self, X_train_raw, show_progress=True):
        # check how many epoch trained before
        with open('./tmp/trained_epoch.txt') as f:
            self.trained_epoch = int(f.readlines()[0])


        if self.core.n_features is None:
            self.core.set_num_features(n_entity+n_relation)

        # assert self.core.n_features == n_relation, 'Different num of features in initialized graph and input'

        if self.core.graph is None:
            if os.path.isfile('./tmp/my-model.meta'):
                self.restore_graph()
            else:
                self.core.build_graph()
                self.initialize_session()

        #if epochs is None:
        #    epochs = self.epochs

        # For reproducible results
        if self.seed:
            np.random.seed(self.seed)

        # Training cycle
        mask = np.random.permutation(X_train_raw.shape[0])
        X_train_shuffled = X_train_raw[mask]
        min_MR = n_entity
        sr_dict = load_obj('sr_dict_double')
        ro_dict = load_obj('ro_dict_double')

        for epoch in tqdm(range(self.epochs), unit='epoch', disable=(not show_progress)):
            epoch_loss_train = []
            epoch_loss_entropy = []
            epoch_loss_reg = []

            # iterate over batches

            for j in range(0, X_train_shuffled.shape[0], self.batch_size):

                lower_bound = j
                upper_bound = min(j + self.batch_size, X_train_shuffled.shape[0])

                n_positive_samples = upper_bound - lower_bound
                n_negative_samples = n_positive_samples * self.negative_sample
                n_samples = n_positive_samples + n_negative_samples

                # generating negative tuples
                subjects = np.zeros(n_samples, dtype=np.int)
                objects = np.zeros(n_samples, dtype=np.int)
                relations = np.zeros(n_samples, dtype=np.int)
                labels = np.zeros(n_samples, dtype=np.int)

                i = 0
                for (s, r, o) in zip(X_train_shuffled[lower_bound:upper_bound, 0], X_train_shuffled[lower_bound:upper_bound, 1], X_train_shuffled[lower_bound:upper_bound, 2]):
                    head = float(len(ro_dict[(r, o)]))
                    tail = float(len(sr_dict[(s, r)]))
                    hpt = head / tail
                    tph = tail / head

                    num_neg_o = max(int(hpt / (hpt + tph) * self.negative_sample), 1)
                    num_neg_s = self.negative_sample - num_neg_o

                    o_selection = np.delete(np.arange(n_entity), sr_dict[(s, r)])
                    neg_o = np.random.choice(o_selection, size=num_neg_o)
                    s_selection = np.delete(np.arange(n_entity), ro_dict[(r, o)])
                    neg_s = np.random.choice(s_selection, size=num_neg_s)

                    p = i * (self.negative_sample + 1)
                    q = (i + 1) * (self.negative_sample + 1)
                    objects[p:q] = np.concatenate([[o], neg_o, np.repeat(o, num_neg_s)])
                    subjects[p:q] = np.concatenate([[s], np.repeat(s, num_neg_o), neg_s])
                    relations[p:q] = np.repeat(r, self.negative_sample + 1)
                    labels[p] = 1
                    i += 1

                # one-hot encoding for s & o
                row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
                column = np.concatenate([subjects, [n_entity - 1]])
                values = np.concatenate([np.ones(n_samples), [0]])
                bXs = csr_matrix((values, (row, column)))
                del row, column, values

                row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
                column = np.concatenate([relations, [n_relation - 1]])
                values = np.concatenate([np.ones(n_samples), [0]])
                bXr = csr_matrix((values, (row, column)))
                del row, column, values

                row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
                column = np.concatenate([objects, [n_entity - 1]])
                values = np.concatenate([np.ones(n_samples), [0]])
                bXo = csr_matrix((values, (row, column)))
                del row, column, values

                bY = labels

                del subjects, objects, relations

                fd = batch_to_feeddict(bXs, bXr, bXo, bY, core=self.core)
                ops_to_run = [self.core.trainer, self.core.summary_op, self.core.checked_target]
                result = self.session.run(ops_to_run, feed_dict=fd)
                _, summary_str, batch_target_value = result
                epoch_loss_train.append(batch_target_value)
                # Calculate the accuracy on the training-batch.
            train_loss = np.mean(epoch_loss_train)

            if self.verbose >= 1:
                print('[epoch {}]: mean target value: {}'.format(epoch+1, train_loss))

            # save model
            if (epoch + 1) % 5 == 0:
                self.core.saver.save(self.session, './tmp/my-model')
                with open('./tmp/trained_epoch.txt', "w") as f:
                    f.write("{0}".format(self.trained_epoch+epoch+1))
                print 'model saved.'

            # write stats
            if self.need_logs:
                self.summary_writer.add_summary(summary_str, self.steps)
                self.summary_writer.flush()
            self.steps += 1


    def decision_function(self, Xsr, Xro, Xso, pred_batch_size):
        if self.core.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        if pred_batch_size is None:
            pred_batch_size = self.batch_size

        for bXsr, bXro, bXso, bY in batcher(Xsr, Xro, Xso, y_=None, batch_size=pred_batch_size):
            fd = batch_to_feeddict(bXsr, bXro, bXso, bY, core=self.core)
            output.append(np.array(self.session.run(self.core.outputs, feed_dict=fd)).flatten())
        distances = np.concatenate(output).flatten()
        # WARN: be careful with this reshape in case of multi-class
        return distances

    @abstractmethod
    def predict(self, X, pred_batch_size=None):
        """Predict target values for X."""

    @property
    def intercept(self):
        """Export bias term from tf.Variable to float."""
        return self.core.b.eval(session=self.session)

    @property
    def weights(self):
        """Export underlying weights from tf.Variables to np.arrays."""
        return [x.eval(session=self.session) for x in self.core.w]

    def destroy(self):
        """Terminates session and destroys graph."""
        self.session.close()
        self.core.graph = None

    def restore_graph(self):
        self.core.graph = tf.Graph()
        self.core.graph.seed = self.seed
        self.session = tf.Session(config=self.session_config, graph=self.core.graph)

        with self.core.graph.as_default():
            self.restorer = tf.train.import_meta_graph('./tmp/my-model.meta')
            self.restorer.restore(self.session, tf.train.latest_checkpoint('./tmp/'))

            # restore ops tensor

            self.core.outputs = tf.get_collection('outputs')[0]
            self.core.checked_target = tf.get_collection('target')[0]
            self.core.trainer = tf.get_collection('trainer')[0]
            self.core.summary_op = tf.get_collection('summary_op')[0]
            self.core.saver = tf.train.Saver()   # cannot store saver in collection, so create a new

            self.core.n_entity = tf.get_collection('n_entity')[0]
            self.core.ranking = tf.py_func(rankdata, inp=[self.core.outputs, 'max'], Tout=tf.int64)
            self.core.ranking = self.core.n_entity + 1 - self.core.ranking


            # restore placeholder tensor

            self.core.xs_raw_indices = tf.get_collection('xs_raw_indices')[0]
            self.core.xs_raw_values = tf.get_collection('xs_raw_values')[0]
            self.core.xs_raw_shape = tf.get_collection('xs_raw_shape')[0]
            self.core.train_xs = tf.SparseTensor(self.core.xs_raw_indices, self.core.xs_raw_values, self.core.xs_raw_shape)

            self.core.xr_raw_indices = tf.get_collection('xr_raw_indices')[0]
            self.core.xr_raw_values = tf.get_collection('xr_raw_values')[0]
            self.core.xr_raw_shape = tf.get_collection('xr_raw_shape')[0]
            self.core.train_xr = tf.SparseTensor(self.core.xr_raw_indices, self.core.xr_raw_values, self.core.xr_raw_shape)

            self.core.xo_raw_indices = tf.get_collection('xo_raw_indices')[0]
            self.core.xo_raw_values = tf.get_collection('xo_raw_values')[0]
            self.core.xo_raw_shape = tf.get_collection('xo_raw_shape')[0]
            self.core.train_xo = tf.SparseTensor(self.core.xo_raw_indices, self.core.xo_raw_values, self.core.xo_raw_shape)

            self.core.train_y = tf.get_collection('Y')[0]

            # restore variables
            self.core.Ve = tf.get_collection('Ve')[0]
            self.core.Vr = tf.get_collection('Vr')[0]
            self.core.Wsr = tf.get_collection('Wsr')[0]
            self.core.Wro = tf.get_collection('Wro')[0]
            self.core.Wso = tf.get_collection('Wso')[0]
            self.core.Wss = tf.get_collection('Wss')[0]
            self.core.Wrr = tf.get_collection('Wrr')[0]
            self.core.Woo = tf.get_collection('Woo')[0]

            self.core.T = tf.get_collection('T')[0]
