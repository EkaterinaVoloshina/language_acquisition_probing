import pandas as pd

def load_files(dataset, path=None):
    senteval = ['subj_number', 'top_const', 'tree_depth']
    if dataset in senteval:
        data = pd.read_csv(path, sep='\t', header=None)
        TRAIN = data[data[0] == 'tr']
        TEST = data[data[0] == 'te']
        X_train = TRAIN[2]
        X_test = TEST[2]
        y_train = TRAIN[1].values
        y_test = TEST[1].values
    elif dataset == 'person':
        data = pd.read_csv(path, sep='\t')
        TRAIN = data[data['subset']=='tr']
        TEST = data[data['subset']=='te']
        X_train = TRAIN['text']
        X_test = TEST['text']
        y_train = TRAIN['label'].values
        y_test = TEST['label'].values
    elif dataset == 'conn':
        TRAIN = pd.read_csv('Conn_train.tsv', sep='\t')
        TEST = pd.read_csv('Conn_test.tsv', sep='\t')
        X_train = TRAIN[['sentence_1', 'sentence_2']].values.tolist()
        X_test = TEST[['sentence_1', 'sentence_2']].values.tolist()
        y_train = TRAIN['marker'].values
        y_test = TEST['marker'].values
    elif dataset == 'DC':
        TRAIN = pd.read_csv('DC_train.csv')
        TEST = pd.read_csv('DC_test.csv')
        X_train = TRAIN['sentence'].apply(eval)
        X_test = TEST['sentence'].apply(eval)
        y_train = TRAIN['label'].values
        y_test = TEST['label'].values
    elif dataset == 'PDTB':
        TRAIN = pd.read_csv('Conn_train.tsv', sep='\t')
        TEST = pd.read_csv('Conn_test.tsv', sep='\t')
        X_train = TRAIN[['sentence_1', 'sentence_2']].values.tolist()
        X_test = TEST[['sentence_1', 'sentence_2']].values.tolist()
        y_train = TRAIN['label'].values
        y_test = TEST['label'].values
    return X_train, y_train, X_test, y_test
