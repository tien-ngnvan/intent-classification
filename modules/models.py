from tensorflow.keras.layers import *
from modules.arc_face import OutputLayer, ArcHead
from tensorflow.keras import Model
from utils import config
from deep_nlp import ner, pos_tagging
from modules.Hierarchical_Attn import AttentionWithContext
from tensorflow.keras import regularizers


def intent_model(num_class):
    label = Input(shape=[], name='label')

    # ======================== NER model =======================================
    pos_model = pos_tagging.build_model('distilroberta-base', 'distilroberta-base',
                                        max_length=config.MAX_SEQUENCE, num_pos=38)
    model = pos_tagging.package_layer_pos(pos_model)
    new_model = ner.package_layer_ner(model, 10, 'adam')
    new_model.load_weights(config.ner_path)
    new_model.trainable = False

    # =============================== INTENT =====================================
    ner_layer = new_model.get_layer('bilstm_ner_2').output
    embds_bert = new_model.get_layer('pretrained_nlp').output['last_hidden_state']
    concat_layer = Concatenate(axis=-1, name='concat_NER')([embds_bert, ner_layer])
    norm_layer = LayerNormalization(epsilon=1e-03, name='norm_intent')(concat_layer)

    biLSTM_1 = Bidirectional(LSTM(384, return_sequences=True,
                                  dropout=0.4, recurrent_dropout=0.4,
                                  kernel_regularizer=regularizers.L2(l2=1e-4),
                                  name='biLSTM_intent1'))(norm_layer)

    biLSTM_2 = Bidirectional(LSTM(256, return_sequences=True,
                                  dropout=0.4, recurrent_dropout=0.4,
                                  kernel_regularizer=regularizers.L2(l2=1e-4),
                                  name='biLSTM_intent2'))(biLSTM_1)
    X = AttentionWithContext()(biLSTM_2)

    # ArcFace
    embds = OutputLayer(512)(X)
    logist = ArcHead(num_classes=num_class, margin=0.5, logist_scale=64,
                     name='archead')(embds, label)

    model = Model(inputs=(new_model.input, label), outputs=[logist, embds],
                  name='DistilRobeta_ArcFace')
    return model


if __name__ == '__main__':
    model = intent_model(21)
    model.summary()



