import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow_addons.layers import GELU
from deep_nlp.pos_tagging import package_layer_pos
from deep_nlp.pos_tagging import build_model
from deep_nlp.CRF import CRF


def ner_layer(last_encode, pos_tag, units, use_dimensional=256):
    soft_embedding = Dense(use_dimensional, use_bias=False, name='soft_embedding_ner_1')(pos_tag)  #
    x = Concatenate(axis=-1, name='concatenate_ner_1')([last_encode, soft_embedding])  # depth > 768
    x = Dense(128, name='dense_ner_1')(x)
    x = LayerNormalization(epsilon=1e-05, name='layernorm_ner_1')(x)
    x = GELU(name='gelu_ner_1')(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1), name='bilstm_ner_1')(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1), name='bilstm_ner_2')(x)
    x = TimeDistributed(Dense(units, activation='relu'), name='timedistributed_ner_1')(x)
    return x


def package_layer_ner(pretrain_pos_model, units, optimizer):
    crf = CRF(units, name='crf_layer')
    model = pretrain_pos_model
    last_encode = model.output[0]
    pos = model.output[1]
    x = ner_layer(last_encode, pos, units)
    x = crf(x)

    new_model = Model(model.inputs, x)

    new_model.compile(optimizer=optimizer, loss={'crf_layer': crf.get_loss}, metrics=[crf.get_accuracy])
    return new_model


