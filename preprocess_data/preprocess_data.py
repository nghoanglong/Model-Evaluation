import numpy as np
import pandas as pd
import os


class PreprocessData:
    def __init__(self,
                 FOLDER_PATH):
        self.folder_path = FOLDER_PATH

    def load_dataset(self,
                     type_dataset):
        if type_dataset == 'train':
            pass
        elif type_dataset == 'dev':
            pass
        else:
            pass

    def SplitToken_FromTreebank(self,
                                treebank):
        pass

    def Tree_toSentence(self,
                        treebank):
        with open('./data/trees/train.txt', 'r') as f:
            li_sentence = []
            for line in f:
                li_sentence.append(line.rstrip('\n'))
        from nltk.tree import Tree
        t = Tree.fromstring(str(li_sentence[0]))
        sent = ' '.join(t.leaves())
        print(sent)

    def assign_sentiment(self,
                         file_phrases,
                         file_sentimentLabels):
        """Gán sentiment labels cho các phrases tương ứng

            Kết quả là 1 DataFrame được lưu thành file csv

            Giá trị trả về là 5 DataFrame được phân loại thành 5 class tương ứng theo giá trị sentiment

                very_negative = [0, 0.2]
                negative = (0.2, 0.4]
                neutral = (0.4, 0.6]
                positive = (0.6, 0.8]
                very_positive = (0.8, 1.0]

                return (very_negative, negative, neutral, positive, very_positive)
        """
        df_phrases = pd.read_csv(file_phrases, sep='|', header=None)
        df_sentimentLabels = pd.read_csv(file_sentimentLabels, sep='|')

        df_phrases.columns = ['phrases', 'phrase ids']

        df_assignLabels = pd.merge(
            df_phrases, df_sentimentLabels, on='phrase ids')
        df_assignLabels = df_assignLabels[[
            'phrase ids', 'phrases', 'sentiment values']]

        # class phrases have very negative sentiment
        very_negative_phrases = df_assignLabels.loc[(df_assignLabels['sentiment values'] >= 0.0) & (
            df_assignLabels['sentiment values'] <= 0.2)]

        # class phrases have negative sentiment
        negative_phrases = df_assignLabels.loc[(df_assignLabels['sentiment values'] > 0.2) & (
            df_assignLabels['sentiment values'] <= 0.4)]

        # class phrases have neutral sentiment
        neutral_phrases = df_assignLabels.loc[(df_assignLabels['sentiment values'] > 0.4) & (
            df_assignLabels['sentiment values'] <= 0.6)]

        # class phrases have positive sentiment
        positive_phrases = df_assignLabels.loc[(df_assignLabels['sentiment values'] > 0.6) & (
            df_assignLabels['sentiment values'] <= 0.8)]

        # class phrases have very positive sentiment
        very_positive_phrases = df_assignLabels.loc[(df_assignLabels['sentiment values'] > 0.8) & (
            df_assignLabels['sentiment values'] <= 1.0)]

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
            return (very_negative_phrases, negative_phrases, neutral_phrases, positive_phrases, very_positive_phrases)

    def split_dataset(self,
                      file_dataSentence,
                      file_labelSplitData):
        """Chia dataset gồm các sentence thành các file data khác nhau dựa theo label, rồi tạo folder để lưu trữ file
        
            label 1 = train data
            label 2 = test data
            label 3 = dev data
        """

        df_dataSentence = pd.read_csv(file_dataSentence, sep='\t')
        df_labelSplitData = pd.read_csv(file_labelSplitData, sep=',')
        df_mergeSplitLabel = pd.merge(df_dataSentence,
                                      df_labelSplitData,
                                      on='sentence_index')

        grp_splitlabel = df_mergeSplitLabel.groupby('splitset_label')
        train_data = grp_splitlabel.get_group(
            1).loc[:, ['sentence', 'splitset_label']]
        test_data = grp_splitlabel.get_group(
            2).loc[:, ['sentence', 'splitset_label']]
        dev_data = grp_splitlabel.get_group(
            3).loc[:, ['sentence', 'splitset_label']]

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
    # data = PreprocessData('./data/ScoreNLP')
    # train_data = data.load_dataset('train')

    # # demo split dataset
    # data.split_dataset('./collect_data/stanfordSentimentTreebank/datasetSentences.txt',
    #                    './collect_data/stanfordSentimentTreebank/datasetSplit.txt')

    # # demo assign sentiment
    # data.assign_sentiment('./stanfordSentimentTreebank/dictionary.txt',
    #                       './stanfordSentimentTreebank/sentiment_labels.txt')
