from transformers import RobertaConfig, TFAutoModel

def convert_tfmodel(dir_pytorch_model, dir_config):
    config = RobertaConfig.from_pretrained(dir_config)
    tf_model = TFAutoModel.from_pretrained(dir_pytorch_model, config = config, from_pt = True)
    tf_model.layers[0].trainable = False

    return tf_model