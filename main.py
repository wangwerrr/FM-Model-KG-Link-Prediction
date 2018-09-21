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
    epochs=5,
    r_rank=2,
    e_rank=5,
    negative_sample=3,
    batch_size=2000,   # number of positive tuple per batch
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    init_std=1,     # stddev of parameters
    reg=0.1,
    input_type='sparse',
    # seed=42,
)
model.fit(train_set_raw[:10000], show_progress=True)

############################################### reading testing data ###########################################
name = ['subject', 'object', 'relation']
data = pd.read_table('./FB15k/test.txt', sep='\t', header=None, names=name, engine='python').head(100)
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

print 'Ranking o...'
MR = []
MRR = []
HIT1 = []
HIT3 = []
HIT10 = []
for (s, r, o) in zip(test_set_raw[:, 0], test_set_raw[:, 1], test_set_raw[:, 2]):
# one-hot encoding for s , adding negative samples per tuple
    row = np.concatenate([np.arange(n_entity), [n_entity-1]])
    column = np.concatenate([np.repeat(s, n_entity), [n_entity-1]])
    values = np.concatenate([np.ones(n_entity), [0]])
    Xs = csr_matrix((values, (row, column)))
    del row, column, values
# make sure the row is n_relation in length
    row = np.concatenate([np.arange(n_entity), [n_entity-1]])
    column = np.concatenate([np.repeat(r, n_entity), [n_relation-1]])
    values = np.concatenate([np.ones(n_entity), [0]])
    Xr = csr_matrix((values, (row, column)))
    del row, column, values
# one-hot encoding for o , adding negative samples per tuple
    row = np.concatenate([np.arange(n_entity), [n_entity-1]])
    column = np.concatenate([np.arange(n_entity), [n_entity-1]])
    values = np.concatenate([np.ones(n_entity), [0]])
    Xo = csr_matrix((values, (row, column)))
    del row, column, values

    predictions = model.predict(Xs, Xr, Xo, pred_batch_size=Xs.shape[0])

    r, rr, h1, h3, h10 = rank(predictions, o, n_entity)
    MR.append(r)
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

print 'MR(o) =', np.mean(MR)
print 'MRR(o) =', np.mean(MRR)
print 'HIT@1(o) =', np.mean(HIT1)
print 'HIT@3(o) =', np.mean(HIT3)
print 'HIT@10(o) =', np.mean(HIT10)
del MR, MRR, HIT1, HIT3, HIT10


print 'Ranking o (with filter)...'
MR = []
MRR = []
HIT1 = []
HIT3 = []
HIT10 = []
# filtering


sr_dict_all = load_obj('sr_dict_all')
ro_dict_all = load_obj('ro_dict_all')

i = 0
for (s, r, o) in zip(test_set_raw[:, 0], test_set_raw[:, 1], test_set_raw[:, 2]):

    neg_o = np.delete(np.arange(n_entity), sr_dict_all[(s, r)])
    n_samples = neg_o.shape[0] + 1

    # one-hot encoding for s & o
    row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
    column = np.concatenate([np.repeat(s, n_samples), [n_entity - 1]])
    values = np.concatenate([np.ones(n_samples), [0]])
    Xs = csr_matrix((values, (row, column)))
    del row, column, values

    row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
    column = np.concatenate([np.repeat(r, n_samples), [n_relation - 1]])
    values = np.concatenate([np.ones(n_samples), [0]])
    Xr = csr_matrix((values, (row, column)))
    del row, column, values

    row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
    column = np.concatenate([[o], neg_o, [n_entity - 1]])
    values = np.concatenate([np.ones(n_samples), [0]])
    Xo = csr_matrix((values, (row, column)))
    del row, column, values

    predictions = model.predict(Xs, Xr, Xo, pred_batch_size=Xs.shape[0])
# the first sample is the positive fact!!!!!!!
    r, rr, h1, h3, h10 = rank(predictions, 0, n_samples)
    MR.append(r)
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

print 'MR(o)-f =', np.mean(MR)
print 'MRR(o)-f =', np.mean(MRR)
print 'HIT@1(o)-f =', np.mean(HIT1)
print 'HIT@3(o)-f =', np.mean(HIT3)
print 'HIT@10(o)-f =', np.mean(HIT10)
del MR, MRR, HIT1, HIT3, HIT10



############# change s
print 'Ranking...(according to s)'
MR = []
MRR = []
HIT1 = []
HIT3 = []
HIT10 = []

for (s, r, o) in zip(test_set_raw[:, 0], test_set_raw[:, 1], test_set_raw[:, 2]):
# one-hot encoding for s , adding negative samples per tuple
    row = np.concatenate([np.arange(n_entity), [n_entity-1]])
    column = np.concatenate([np.arange(n_entity), [n_entity-1]])
    values = np.concatenate([np.ones(n_entity), [0]])
    Xs = csr_matrix((values, (row, column)))
    del row, column, values
# make sure the row is n_relation in length
    row = np.concatenate([np.arange(n_entity), [n_entity-1]])
    column = np.concatenate([np.repeat(r, n_entity), [n_relation-1]])
    values = np.concatenate([np.ones(n_entity), [0]])
    Xr = csr_matrix((values, (row, column)))
    del row, column, values
# one-hot encoding for o , adding negative samples per tuple
    row = np.concatenate([np.arange(n_entity), [n_entity-1]])
    column = np.concatenate([np.repeat(o, n_entity), [n_entity-1]])
    values = np.concatenate([np.ones(n_entity), [0]])
    Xo = csr_matrix((values, (row, column)))
    del row, column, values

    predictions = model.predict(Xs, Xr, Xo, pred_batch_size=Xs.shape[0])

    r, rr, h1, h3, h10 = rank(predictions, s, n_entity)
    MR.append(r)
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

print 'MR(s) =', np.mean(MR)
print 'MRR(s) =', np.mean(MRR)
print 'HIT@1(s) =', np.mean(HIT1)
print 'HIT@3(s) =', np.mean(HIT3)
print 'HIT@10(s) =', np.mean(HIT10)
del MR, MRR, HIT1, HIT3, HIT10



print 'Ranking (with filter)(according to s)...'
MR = []
MRR = []
HIT1 = []
HIT3 = []
HIT10 = []
# filtering


i = 0
for (s, r, o) in zip(test_set_raw[:, 0], test_set_raw[:, 1], test_set_raw[:, 2]):

    neg_s = np.delete(np.arange(n_entity), ro_dict_all[(r, o)])

    n_samples = neg_s.shape[0] + 1
    # one-hot encoding for s & o
    row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
    column = np.concatenate([[s], neg_s, [n_entity - 1]])
    values = np.concatenate([np.ones(n_samples), [0]])
    Xs = csr_matrix((values, (row, column)))
    del row, column, values

    row = np.concatenate([np.arange(n_samples), [n_samples - 1]])
    column = np.concatenate([np.repeat(r, n_samples), [n_relation - 1]])
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
    r, rr, h1, h3, h10 = rank(predictions, 0, n_samples)
    MR.append(r)
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

print 'MR(s)-f =', np.mean(MR)
print 'MRR(s)-f =', np.mean(MRR)
print 'HIT@1(s)-f =', np.mean(HIT1)
print 'HIT@3(s)-f =', np.mean(HIT3)
print 'HIT@10(s)-f =', np.mean(HIT10)
del MR, MRR, HIT1, HIT3, HIT10

model.destroy()
