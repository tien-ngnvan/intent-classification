import tensorflow as tf
import pandas as pd
import argparse
import os
import sys
import matplotlib.pyplot as plt

from plot_keras_history import plot_history
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_helper import DataLoader
from modules import models
from utils import softmax_loss, optimizer


def callbacks(path_ckpt):
    cb = []
    checkpoint = ModelCheckpoint(path_ckpt, monitor='val_loss', mode='auto',
                                 save_best_only=True, save_weights_only=True)
    cb.append(checkpoint)
    er = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, mode='min',
                        restore_best_weights=True)
    cb.append(er)
    return cb

def main():
    # path dataset
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_train', type=str, default='./data/original/clinc150/train_clinc150.csv',
                        help='train filepath')
    parser.add_argument('--path_val', type=str, default='./data/original/clinc150/val_clinc150.csv',
                        help='val filepath')
    parser.add_argument('--type_model_hg', type=str, default='distilroberta-base',
                        help='Pretrain with model Transformer')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs')
    parser.add_argument('--lr', type=float, default=0.0006,
                        help='learning_rate')
    parser.add_argument('--maxlen', type=int, default=128,
                        help='max_length')
    parser.add_argument('--path_ckpt', type=str, default='../intent.h5',
                        help='path_save_model')
    parser.add_argument('--path_history', type=str, default='../model.png',
                        help='path_history_model')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    return args

def training():
    args = main()

    #   Get data
    data_train = pd.read_csv(args.path_train, encoding='latin-1')
    data_val = pd.read_csv(args.path_val, encoding='latin-1')

    input_train, label_train = DataLoader(data_train, args.maxlen, 'clinc150',
                                        args.type_model_hg).get_data()

    input_val, label_val = DataLoader(data_val, args.maxlen, 'clinc150',
                                        args.type_model_hg).get_data()

    #  Build model
    model = models.build_model(num_classes=150)
    model.summary()

    # get status ckpt: available or None
    # checkpoint path = config.intent_path || checkpoint_dir = os.path.dirname(config.intent_path)
    status = tf.train.get_checkpoint_state(os.path.dirname(args.path_ckpt))
    if status and status.model_checkpoint_path:
      print("Reload checkpoint for training model")
      model.load_weights(args.path_ckpt).expect_partial()
    else:
      print("Training from scratch")

    model.compile(optimizer= optimizer(args.lr, args.epochs, args.batch_size, len(label_train)),
                  loss= {'archead': softmax_loss})

    history = model.fit(x=(input_train, label_train), y= label_train,
                        validation_data= ((input_val, label_val),label_val),
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        callbacks=callbacks(args.path_ckpt),
                        shuffle=True)

    plot_history(history, path=args.path_history)


    return history


if __name__ == '__main__':
    history = training()
