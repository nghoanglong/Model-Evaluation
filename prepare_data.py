import numpy as np
import pandas as pd
import os

def split_dataset(file_dataSentence, file_labelSplitData):
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
    train_data = grp_splitlabel.get_group(1).loc[:, ['sentence', 'splitset_label']]
    test_data = grp_splitlabel.get_group(2).loc[:, ['sentence', 'splitset_label']]
    dev_data = grp_splitlabel.get_group(3).loc[:, ['sentence', 'splitset_label']]
    
    datasets = np.array(
        [train_data, test_data, dev_data], 
        dtype=object
    )

    for file in datasets:
        file.index = pd.MultiIndex.from_arrays([np.arange(len(file.index))], names=['index'])
    
    # create and save data to folder
    try: 
        path = os.getcwd() + "\data"
        if os.path.exists(path):
            raise OSError
        else:
            os.mkdir(path)
    except OSError:
        print (f"Can't create folder at {path}")
    else:
        print (f"Successfully created folder at {path}")
        train_data.to_csv(path + "\\train_data.csv")
        dev_data.to_csv(path + "\\dev_data.csv")
        test_data.to_csv(path + "\\test_data.csv")


if __name__ == '__main__':
    # demo split dataset
    split_dataset('./stanfordSentimentTreebank/datasetSentences.txt',
                './stanfordSentimentTreebank/datasetSplit.txt')