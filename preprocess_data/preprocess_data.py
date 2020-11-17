import numpy as np
import pandas as pd
import os
from nltk.tree import Tree
from nltk.tokenize import word_tokenize


class PreprocessData:
    def __init__(self,
                 FOLDER_PATH):
        """Constructor với tham số nhận vào làm FOLDER_PATH

            FOLDER_PATH định dạng ví dụ: dir_current/data_set/
        """
        self.folder_path = FOLDER_PATH
        self.dataset = None
        self.lib_tokens = None

    def load_dataset(self,
                     type_dataset,
                     file_extension='.txt'):
        """Load dataset tương ứng theo type_dataset

            type_dataset = [train, dev, test] | type = string
            file_extension = .txt, .csv,... | type = string

            return ndarray shape(n, 1) với
                row = sentence được format theo treebank
        """
        try:
            DATASET_REQ_PATH = self.folder_path + '/' + type_dataset + file_extension
            check_exist = os.path.isfile(DATASET_REQ_PATH)
            if check_exist:
                with open(DATASET_REQ_PATH, 'r') as reader:
                    self.dataset = np.array(
                        [line.rstrip("\n") for line in reader])
                return self.dataset
            else:
                raise FileExistsError('File nay ko ton tai')
        except FileExistsError as err:
            print(err)
            return None

    def SplitToken_FromTreebank(self,
                                treebank):
        """Split list các token từ TreeBank
            
            treebank - type string
            return array = [token, token, token,...]
        """
        tree = Tree.fromstring(str(treebank))
        return tree.leaves()

    def Tree_toSentence(self,
                        treebank):
        """Convert tree thành một sentence hoàn chỉnh với

            treebank - type string
            return sentence
        """
        sentence = ' '.join(self.SplitToken_FromTreebank(treebank))
        return sentence

    def getAllTokens(self):
        """Tạo một kho các tokens từ list các sentences

            return dictionary{token: ids, token: ids,...}
        """
        big_sent = ' '.join(self.Tree_toSentence(sentence) for sentence in self.dataset)
        li_tokens = word_tokenize(big_sent)
        self.lib_tokens = {}
        ids = 1
        for token in li_tokens:
            if token not in self.lib_tokens:
                self.lib_tokens[token] = ids
                ids = ids + 1
        return self.lib_tokens
        

    def assign_sentiment(self,
                         file_phrases,
                         file_sentimentLabels):
        """Gán sentiment labels cho các phrases tương ứng

            Kết quả là 1 DataFrame được lưu thành file csv với định dạng
                row = samples
                column = ['phrase ids', 'phrases', 'sentiment values']

        """
        df_phrases = pd.read_csv(file_phrases, sep='|', header=None)
        df_sentimentLabels = pd.read_csv(file_sentimentLabels, sep='|')

        df_phrases.columns = ['phrases', 'phrase ids']

        df_assignLabels = pd.merge(
            df_phrases, df_sentimentLabels, on='phrase ids')
        df_assignLabels = df_assignLabels[[
            'phrase ids', 'phrases', 'sentiment values']]

        # create and save data to folder
        try:
            path = os.getcwd() + "\data"
            if os.path.exists(path):
                raise OSError
            else:
                os.mkdir(path)
        except OSError:
            print(f"Can't create folder at {path} because it was existed")
        else:
            print(f"Successfully created folder at {path}")
        finally:
            print(f'Create and save file data at {path}')
            df_assignLabels.to_csv(path + "\\phrases_and_sentiment.csv")

    def split_dataset(self,
                      SENTENCE_PATH,
                      LABEL_SENTENCE_PATH):
        """Chia dataset gồm các sentence thành các file data khác nhau dựa theo label, rồi tạo folder để lưu trữ file
        
            label 1 = train data
            label 2 = test data
            label 3 = dev data
        """

        df_dataSentence = pd.read_csv(SENTENCE_PATH, sep='\t')
        df_labelSplitData = pd.read_csv(LABEL_SENTENCE_PATH, sep=',')
        df_mergeSplitLabel = pd.merge(df_dataSentence,
                                      df_labelSplitData,
                                      on='sentence_index')

        grp_splitlabel = df_mergeSplitLabel.groupby('splitset_label')
        train_data = grp_splitlabel.get_group(1).loc[:, ['sentence', 'splitset_label']]
        test_data = grp_splitlabel.get_group(2).loc[:, ['sentence', 'splitset_label']]
        dev_data = grp_splitlabel.get_group(3).loc[:, ['sentence', 'splitset_label']]

        datasets = np.array(
            [train_data, test_data, dev_data],
            dtype=object
        )

        # thêm cột index [0, 1, 2, 3,...]
        for file in datasets:
            file.index = pd.MultiIndex.from_arrays(
                [np.arange(len(file.index))], names=['index'])

        # create and save data to folder
        try:
            path = os.getcwd() + "\data"
            if os.path.exists(path):
                raise OSError
            else:
                os.mkdir(path)
        except OSError:
            print(f"Can't create folder at {path} because it was existed")
        else:
            print(f"Successfully created folder at {path}")
        finally:
            print(f'Create and Save file data at {path}')
            train_data.to_csv(path + "\\train_data.csv")
            dev_data.to_csv(path + "\\dev_data.csv")
            test_data.to_csv(path + "\\test_data.csv")

    

if __name__ == '__main__':
    # demo load dataset
    data = PreprocessData('./data/trees')
    train_data = data.load_dataset('train', '.txt')

    lib_tokens = data.getAllTokens()
    print(len(lib_tokens))

    
