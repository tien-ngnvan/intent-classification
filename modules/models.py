from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import regularizers

from modules.arc_face import OutputLayer, ArcHead
from deep_nlp import ner, pos_tagging
from modules.Hierarchical_Attn import AttentionWithContext

PATH_NER = '/content/drive/MyDrive/intent/weights/nertagl.h5'

def build_model(num_classes):
    label = Input(shape=[], name='label')

    # ======================== NER model =======================================
    pos_model = pos_tagging.build_model('distilroberta-base', 'distilroberta-base',
                                        max_length=128, num_pos=38)
    model = pos_tagging.package_layer_pos(pos_model)
    new_model = ner.package_layer_ner(model, 10, 'adam')
    new_model.load_weights(PATH_NER)
    new_model.trainable = False

    # =============================== INTENT ===================================
    ner_layer = new_model.get_layer('bilstm_ner_2').output
    embds_bert = new_model.get_layer('pretrained_nlp').output['last_hidden_state']
    concat_layer = Concatenate(axis=-1, name='concat_NER')([embds_bert, ner_layer])
    norm_layer = LayerNormalization(epsilon=1e-03, name='norm_intent')(concat_layer)

    biLSTM_1 = Bidirectional(LSTM(320, return_sequences=True,
                                  dropout=0.5, recurrent_dropout=0.5,
                                  kernel_regularizer=regularizers.L2(l2=1e-4),
                                  name='biLSTM_intent1'))(norm_layer)

    biLSTM_2 = Bidirectional(LSTM(192, return_sequences=True,
                                  dropout=0.5, recurrent_dropout=0.5,
                                  kernel_regularizer=regularizers.L2(l2=1e-4),
                                  name='biLSTM_intent2'))(biLSTM_1)
    X = AttentionWithContext()(biLSTM_2)

    # ArcFace
    embds = OutputLayer(320)(X)
    logist = ArcHead(num_classes=num_classes, margin=0.5, logist_scale=64,
                     name='archead')(embds, label)

    model = Model(inputs=(new_model.input, label), outputs=[logist, embds],
                  name='DistilRobeta_ArcFace')
    return model

if __name__ == '__main__':
    model = build_model(150)
    model.summary()



