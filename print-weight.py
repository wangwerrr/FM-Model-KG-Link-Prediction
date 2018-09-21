from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from fm.models import TFFMClassifier

model = TFFMClassifier(
    epochs=0,
    r_rank=10,
    e_rank=30,
    negative_sample=300,
    batch_size=2000,   # number of positive tuple per batch
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    init_std=1,     # stddev of parameters
    reg=0.1,
    input_type='sparse',
    # seed=42,
)

model.restore_graph()
variables_names = [model.core.Wsr, model.core.Wro, model.core.Wso, model.core.Wss, model.core.Woo, model.core.Wrr]
values = model.session.run(variables_names)
for k, v in zip(variables_names, values):
    print(k, v)
    print('mean:')
    print(np.mean(v))
    print('absolute mean:')
    print(np.mean(np.absolute(v)))