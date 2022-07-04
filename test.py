import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import time

from modules import models
from data_helper import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_test', type=str,
                        default='/content/drive/MyDrive/intent/data/expend/clinc150/test_clinc155.csv',
                        help='test file')
    parser.add_argument('--path_train', type=str,
                        default='/content/drive/MyDrive/intent/data/original/clinc150/train_clinc150.csv',
                        help='train file')
    parser.add_argument('--type_model_hg', type=str, default='distilroberta-base',
                        help='Pretrain with model Transformer')
    parser.add_argument('--knn', type=int, default=2,
                        help='k-Nearest Neighbor')
    parser.add_argument('--maxlen', type=int, default=128,
                        help='max_length')
    parser.add_argument('--path_ckpt', type=str, default='/content/drive/MyDrive/intent/weights/img_eps8_bch32.h5',
                        help='path_save_model')
    parser.add_argument('--dataset', type=str, default='clinc150',
                        help='type dataset')
    args = parser.parse_args()

    # get data
    data_test = pd.read_csv(args.path_test, encoding='latin-1')
    data_train = pd.read_csv(args.path_train, encoding='latin-1')
    if args.dataset == 'clinc150':
        num_cls = 150
    else:
        num_cls = 77

    input_train, label_train = DataLoader(data_train, args.maxlen, 'clinc150',
                                          args.type_model_hg).get_data()
    input_test, label_test = DataLoader(data_test, args.maxlen, 'clinc150',
                                        args.type_model_hg).get_data()

    # create model
    start = time.time()
    model = models.build_model(num_cls)
    print("build model: ", time.time() - start)
    model.load_weights(args.path_ckpt)

    # get embds train
    y_train = np.zeros(len(label_train))
    y_train = tf.convert_to_tensor(y_train)
    _, embds_train = model.predict(x=(input_train, y_train))  # (batch, 320)

    with open('embds_train.npy', 'wb') as f:
        np.save(f, embds_train)
    with open('label_train.npy', 'wb') as f:
        np.save(f, label_train)

    # with open('/content/drive/MyDrive/intent/data/embds_train.npy', 'rb') as f:
    #    embds_train = np.load(f)
    # with open('/content/drive/MyDrive/intent/data/label_train.npy', 'rb') as f:
    #    label_train = np.load(f)
    embds_train = tf.math.l2_normalize(embds_train, axis=1)

    # get embds test
    y_test = np.zeros(len(label_test))
    y_test = tf.convert_to_tensor(y_test)
    _, embds_test = model.predict(x=(input_test, y_test))
    embds_test = tf.math.l2_normalize(embds_test, axis=1)

    # Create KNN
    for i in range(1, 6):
        A = []
        print("\n\n================== KNN: {} =========================".format(i))
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(embds_train, label_train)  # training KNN with embds_train 150
        start = time.time()
        y_pred = neigh.predict(embds_test)  # predict on test set 155
        print("Train_score: ", (neigh.score(embds_train, label_train)))
        print("Test_score: ", (neigh.score(embds_test, label_test)))
        print(classification_report(label_test, y_pred))
        # store in current directory
        # np.savetxt( "a{}.csv".format(i), A, delimiter="," )










