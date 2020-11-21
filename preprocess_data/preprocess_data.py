import numpy as np
import pandas as pd
import os
from nltk.tree import Tree
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class PreprocessData:
    def __init__(self,
                 FOLDER_PATH):
        """Constructor với tham số nhận vào làm FOLDER_PATH

            FOLDER_PATH định dạng ví dụ: dir_current/data_set/
        """
        self.folder_path = FOLDER_PATH

    def load_dataset(self,
                     type_dataset,
                     file_extension='.txt'):
        """Load dataset tương ứng theo type_dataset

            type_dataset = [train, dev, test] | type = string
            file_extension = .txt, .csv,... | type = string

            return ndarray shape(n, 2) với
                row = np.array[list_tokens extract from sentence, label]
        """
        try:
            DATASET_REQ_PATH = self.folder_path + '/' + type_dataset + file_extension
            check_exist = os.path.isfile(DATASET_REQ_PATH)
            if check_exist:
                with open(DATASET_REQ_PATH, 'r') as reader:
                    dataset = np.array([
                        np.array([self.PTB_tokenize(line.rstrip("\n")),
                                  self.PTB_get_label(line)], dtype=object)
                        for line in reader])
                return dataset
            else:
                raise FileExistsError('File nay ko ton tai')
        except FileExistsError as err:
            print(err)
            return None

    def PTB_get_label(self,
                      treebank):
        """get label của root sentece trong PTB
            treebank - type string
            return label
        """
        tree = Tree.fromstring(treebank)
        return tree.label()

    def PTB_tokenize(self,
                     treebank):
        """Split list các token từ cây PTB
            
            treebank - type string
            return array = [token, token, token,...]
        """
        tree = Tree.fromstring(str(treebank))
        return np.array(tree.leaves())

    def transfrom_sentence(self,
                           li_tokens):
        """Transfrom list các tokens thành 1 sentence hoàn chỉnh

            li_tokens = [token, token, token,...]
            return sentence
        """
        sentence = ' '.join(li_tokens)
        return sentence

    def get_list_vocabularies(self,
                              dataset,
                              reverse=False):
        """Tạo một kho các tokens từ list các sentences
            
            dataset = [[token, token, token,...], label]
            return dictionary{token: ids, token: ids,...} if reverse = False
            return dictionary{ids: token, ids: token,...} if reverse = True
        """
        li_tokens = set()
        for sample in dataset:
            li_tokens.update(sample[0])
        
        if reverse:
            lib_tokens = dict([(idx, token)
                            for idx, token in enumerate(li_tokens)])
        else:
            lib_tokens = dict([(token, idx)
                            for idx, token in enumerate(li_tokens)])
        return lib_tokens

    def encode_sentence(self,
                        sent_tokenized,
                        li_vocabs):
        """Encode một sentence về dạng mỗi token tương ứng với một id trong list vocabs

            sent_tokenized - sentence đã được tokenize thành list các tokens
            li_vocabs = {token: id, token: id,...}
            return sentence = [id, id, id, id,...]
        """

        res_encode = np.array([li_vocabs[token] if token in li_vocabs
                               else print('does not have token in li vocabs')
                               for token in sent_tokenized])
        return res_encode

    def decode_sentence(self,
                        li_vocabs):
        """Decode một list id về dạng list các token tương ứng

            sent_tokenized - sentence đã được tokenize thành list các tokens
            li_vocabs = {token: id, token: id,...}
            return sentence = [id, id, id, id,...]
        """
        pass


if __name__ == '__main__':
    # demo
    data = PreprocessData('./data/trees')
    train_data = data.load_dataset('train', '.txt')

    li_vocabs = data.get_list_vocabularies(train_data)  # lấy ra list các vocabs
    encode_li_sent = np.array([torch.tensor(data.encode_sentence(sample, li_vocabs))
                               for sample in train_data[:, 0]],
                              dtype=object)  # encode list các sentence

    padded = pad_sequence(encode_li_sent, batch_first=True)
