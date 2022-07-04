import tensorflow as tf
import pandas as pd
import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_helper import DataLoader
from modules import models
from utils import softmax_loss, optimizer


def callbacks(path_ckpt):
    cb = []
    checkpoint = ModelCheckpoint(path_ckpt, monitor='val_loss', mode='auto',
                                 save_best_only=True, verbose=1)
    cb.append(checkpoint)
    er = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=3,mode='auto')
    cb.append(er)
    return cb

def main():
    # path dataset
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_train_seq_in', type=str, default='./ATIS_VI/train/seq.in',
                        help='train_seq.in')
    parser.add_argument('--path_label', type=str, default='./ATIS_VI/train/label',
                        help='label')
    parser.add_argument('--tokenizer_dir_config', type=str, default='roberta-base',
                        help='Pretrain with model Transformer')
    parser.add_argument('--l2', type=float, default=0.0001,
                        help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout layer')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='epochs')
    parser.add_argument('--lr', type=int, default=0.004,
                        help='learning_rate')
    parser.add_argument('--maxlen', type=int, default=50,
                        help='max_length')
    parser.add_argument('--path_ckpt', type=str, default='./BKAI/ckpt/bkai.h5',
                        help='path_save_ckpt')
    parser.add_argument('--path_log', type=str, default='./BKAI/logs/log',
                        help='path_save_tensorboard')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

def training(args):
    print(args)

    #   Load data
    data_train = pd.read_csv(cfg.train_path, encoding='latin-1')
    data_val = pd.read_csv(cfg.val_path, encoding='latin-1')
    # get Train data
    df_train = DataLoader(data_train, 128, 'distilroberta-base', 'clinc150')
    input_train, label_train = df_train.convert_to_tensor()

    df_val = DataLoader(data_val, 128, 'distilroberta-base', 'clinc150')
    input_val, label_val = df_val.convert_to_tensor()

    #   Build model
    model = models.intent_model(cfg.num_classes)
    model.summary()

    # get status ckpt: available or None
    # checkpoint path = config.intent_path || checkpoint_dir = os.path.dirname(config.intent_path)
    status = tf.train.get_checkpoint_state(os.path.dirname(cfg.intent_path))
    if status and status.model_checkpoint_path:
      print("Reload checkpoint for training model")
      model.load_weights(cfg.intent_path).expect_partial()
    else:
      print("Training from scratch")


    model.compile(optimizer= optimizer(cfg.lr, cfg.epochs, cfg.batch_size, len(label_train)),
                  loss= {'archead': softmax_loss})

    history = model.fit(x=(input_train, label_train), y= label_train,
                        validation_data= ((input_val, label_val),label_val),
                        batch_size=cfg.batch_size,
                        epochs=cfg.epochs,
                        verbose=1,
                        callbacks=callbacks(cfg.intent_path))
    return history
if __name__ == '__main__':
    history = training()
