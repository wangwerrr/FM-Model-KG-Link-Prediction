from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from scipy.sparse import csr_matrix, coo_matrix, hstack

from fm.models import TFFMClassifier
from fm.utils import rank, load_obj

n_entity = 14951
n_relation = 1345


# label encoding with regard to entity2id & relation2id
name = ['entity', 'id']
entity_id = pd.read_table('./FB15k/entities.txt', sep='\t', header=None, names=name, engine='python')
entity = entity_id['entity'].values.tolist()
le_entity = preprocessing.LabelEncoder()
le_entity.fit(entity)

name = ['relation', 'id']
relation_id = pd.read_table('./FB15k/relations.txt', sep='\t', header=None, names=name, engine='python')
relation = relation_id['relation'].values.tolist()
le_relation = preprocessing.LabelEncoder()
le_relation.fit(relation)


#################################### reading training data ###########################################
name = ['subject', 'object', 'relation']
data = pd.read_table('./FB15k/train.txt', sep='\t', header=None, names=name, engine='python')
print 'Loading training data...'

subjects = data['subject'].values.tolist()
objects = data['object'].values.tolist()
relations = data['relation'].values.tolist()


# string list to int array
subjects = np.array(le_entity.transform(subjects))
objects = np.array(le_entity.transform(objects))
relations = np.array(le_relation.transform(relations))
train_objects = objects

train_set_raw = np.concatenate([subjects.reshape(-1, 1), relations.reshape(-1, 1), objects.reshape(-1, 1)], axis=1)
n_pos_samples = len(relations)
print 'num of positive training tuple:', n_pos_samples

del name, data, subjects, objects, relations, n_pos_samples


model = TFFMClassifier(
    epochs=0,
    r_rank=100,
    e_rank=300,
    negative_sample=300,
    batch_size=2000,   # number of positive tuple per batch
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    init_std=1,     # stddev of parameters
    reg=0.1,
    input_type='sparse',
    # seed=42,
)
model.fit(train_set_raw, show_progress=True)

############################################### reading testing data ###########################################
name = ['subject', 'object', 'relation']
data = pd.read_table('./FB15k/test.txt', sep='\t', header=None, names=name, engine='python')
print 'Loading testing data...'

subjects = data['subject'].values.tolist()
objects = data['object'].values.tolist()
relations = data['relation'].values.tolist()


# string list to int array
subjects = np.array(le_entity.transform(subjects))
objects = np.array(le_entity.transform(objects))
relations = np.array(le_relation.transform(relations))
test_objects = objects

test_set_raw = np.concatenate([subjects.reshape(-1, 1), relations.reshape(-1, 1), objects.reshape(-1, 1)], axis=1)
n_pos_samples = test_set_raw.shape[0]
del subjects, objects, relations


print 'Ranking r...'
MR = []
MRR = []
HIT1 = []
HIT3 = []
HIT10 = []
for (s, r, o) in zip(test_set_raw[:, 0], test_set_raw[:, 1], test_set_raw[:, 2]):
# one-hot encoding for s , adding negative samples per tuple
    row = np.concatenate([np.arange(n_relation), [n_relation-1]])
    column = np.concatenate([np.repeat(s, n_relation), [n_entity-1]])
    values = np.concatenate([np.ones(n_relation), [0]])
    Xs = csr_matrix((values, (row, column)))
    del row, column, values
# make sure the row is n_relation in length
    row = np.concatenate([np.arange(n_relation), [n_relation-1]])
    column = np.concatenate([np.arange(n_relation), [n_relation-1]])
    values = np.concatenate([np.ones(n_relation), [0]])
    Xr = csr_matrix((values, (row, column)))
    del row, column, values
# one-hot encoding for o , adding negative samples per tuple
    row = np.concatenate([np.arange(n_relation), [n_relation-1]])
    column = np.concatenate([np.repeat(o, n_relation), [n_entity-1]])
    values = np.concatenate([np.ones(n_relation), [0]])
    Xo = csr_matrix((values, (row, column)))
    del row, column, values

    predictions = model.predict(Xs, Xr, Xo, pred_batch_size=Xs.shape[0])

    rk, rr, h1, h3, h10 = rank(predictions, r, n_relation)
    MR.append(rk)
    MRR.append(rr)
    HIT1.append(h1)
    HIT3.append(h3)
    HIT10.append(h10)

    if len(MRR) % 10000 == 0:
        print len(MRR), '/', n_pos_samples

    del Xs, Xr, Xo

MR = np.array(MR)
MRR = np.array(MRR)
HIT1 = np.array(HIT1)
HIT3 = np.array(HIT3)
HIT10 = np.array(HIT10)

print 'MR(r) =', np.mean(MR)
print 'MRR(r) =', np.mean(MRR)
print 'HIT@1(r) =', np.mean(HIT1)
print 'HIT@3(r) =', np.mean(HIT3)
print 'HIT@10(r) =', np.mean(HIT10)
del MR, MRR, HIT1, HIT3, HIT10


print 'Ranking r (with filter)...'
MR = []
MRR = []
HIT1 = []
HIT3 = []
HIT10 = []


# filtering
so_dict_all = load_obj('so_dict_all')

i = 0
for (s, r, o) in zip(test_set_raw[:, 0], test_set_raw[:, 1], test_set_raw[:, 2]):
    if (s, o) in so_dict_all:
        neg_r = np.delete(np.arange(n_relation), so_dict_all[(s, o)])
    else:
        neg_r = np.delete(np.arange(n_relation), [r])
    n_samples = neg_r.shape[0] + 1

    # one-hot encoding for s & o
    row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
    column = np.concatenate([np.repeat(s, n_samples), [n_entity - 1]])
    values = np.concatenate([np.ones(n_samples), [0]])
    Xs = csr_matrix((values, (row, column)))
    del row, column, values

    row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
    column = np.concatenate([[r], neg_r, [n_relation - 1]])
    values = np.concatenate([np.ones(n_samples), [0]])
    Xr = csr_matrix((values, (row, column)))
    del row, column, values

    row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
    column = np.concatenate([np.repeat(o, n_samples), [n_entity - 1]])
    values = np.concatenate([np.ones(n_samples), [0]])
    Xo = csr_matrix((values, (row, column)))
    del row, column, values

    predictions = model.predict(Xs, Xr, Xo, pred_batch_size=Xs.shape[0])
# the first sample is the positive fact!!!!!!!
    rk, rr, h1, h3, h10 = rank(predictions, 0, n_samples)
    MR.append(rk)
    MRR.append(rr)
    HIT1.append(h1)
    HIT3.append(h3)
    HIT10.append(h10)

    if len(MRR) % 10000 == 0:
        print len(MRR), '/', n_pos_samples

    del Xs, Xr, Xo

MR = np.array(MR)
MRR = np.array(MRR)
HIT1 = np.array(HIT1)
HIT3 = np.array(HIT3)
HIT10 = np.array(HIT10)

print 'MR(r)-f =', np.mean(MR)
print 'MRR(r)-f =', np.mean(MRR)
print 'HIT@1(r)-f =', np.mean(HIT1)
print 'HIT@3(r)-f =', np.mean(HIT3)
print 'HIT@10(r)-f =', np.mean(HIT10)
del MR, MRR, HIT1, HIT3, HIT10


model.destroy()
