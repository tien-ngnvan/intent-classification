import tensorflow as tf
import numpy as np

from transformers import AutoTokenizer


class DataLoader():
    def __init__(self, dataFrame,
                 max_length=128,
                 name_data = 'clinc150',
                 name_pretrain = 'distilroberta-base',):
        """
        :param dataFrame: a dataset has two columns
        :param max_length: the longest of sentence make tokenizer
        :param name_pretrain: pretrain model hugging face
        :param name_data: name of dataset
        """
        assert (name_data not in ['clinc50', 'banking77'])

        self.dataFrame = dataFrame
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name_pretrain)
        self.name_data = name_data

    def encode_token(self, sentence):
        token = self.tokenizer.encode_plus(sentence, truncation=True,
                                           max_length=self.max_length, padding='max_length',
                                           add_special_tokens=True, return_tensors='tf')
        return token['input_ids'][0], token['attention_mask'][0]

    def get_data(self):
        X, Y = [], []
        self.dataFrame = (self.dataFrame.sample(frac=1)).reset_index(drop=True) # shuffle data
        if self.name_data == 'clinc150':
            # text
            for index, item in enumerate(self.dataFrame['sentences']):
                input_ids, attention_mask = self.encode_token(item)
                X.append(input_ids)
                Y.append(attention_mask)
            # label
            facts = np.unique(np.unique(self.dataFrame[['labels']]), return_index=True)
            mapping = dict(zip(*facts))
            self.dataFrame['index'] = self.dataFrame['labels'].map(mapping)

        elif self.name_data == 'banking77':
            # text
            for index, item in enumerate(self.dataFrame['text']):
                input_ids, attention_mask = self.encode_token(item)
                X.append(input_ids)
                Y.append(attention_mask)

            # label
            facts = np.unique(np.unique(self.dataFrame[['category']]), return_index=True)
            mapping = dict(zip(*facts))
            self.dataFrame['index'] = self.dataFrame['category'].map(mapping)

        inputs = (tf.convert_to_tensor(X), tf.convert_to_tensor(Y))
        labels = tf.convert_to_tensor(self.dataFrame['index'])

        return inputs, labels