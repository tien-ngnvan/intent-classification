import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow_addons.layers import GELU
from deep_nlp.load_pretrained import convert_tfmodel


def pos_layer(x, units):
    x = Bidirectional(LSTM(256, return_sequences= True), name= 'bilstm_pos_1')(x)
    x = Bidirectional(LSTM(256, return_sequences= True), name= 'bilstm_pos_2')(x)
    x = Dropout(0.1, name= 'dropout_pos_1')(x)
    x = TimeDistributed(Dense(256), name= 'timedistributed_pos_1')(x)
    x = LayerNormalization(epsilon= 1e-05, name= 'layernorm_pos_1')(x)
    x = GELU(name= 'gelu_pos_1')(x)
    x = Dropout(0.1, name= 'dropout_pos_2')(x)
    x = Dense(units, name= 'dense_pos_1')(x)
    x = Softmax(name= 'softmax_pos_1')(x)
    return x


def build_model(dir_pt_model, dir_config, max_length, num_pos):
    """
    input:
    max_length: length
    """
    bert_pretrained_tf = convert_tfmodel(dir_pytorch_model= dir_pt_model, dir_config= dir_config)
    bert_pretrained_tf._name = "pretrained_nlp"

    input_ids = Input(shape = (max_length), dtype= tf.int32, name= "input_ids")
    attention_mask = Input(shape = (max_length), dtype= tf.int32, name= "attention_mask")
    last_encode = bert_pretrained_tf(input_ids, attention_mask)['last_hidden_state'] # shape (None, length , 768)
    pos = pos_layer(last_encode, num_pos)
    model = Model([input_ids, attention_mask], [pos], name= "pos_tagging")
    return model


def package_layer_pos(pretrain_pos_model):
    model = pretrain_pos_model
    model.trainable = False
    input = model.inputs
    last_encode = model.get_layer('pretrained_nlp').output['last_hidden_state']
    pos_out = model.output

    model_temp = Model(input, [last_encode, pos_out], name= 'multitask_pretrain')
    return model_temp

