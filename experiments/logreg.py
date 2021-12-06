import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


class LogRegClassification(object):

    def __init__(self, train_y, test_y, checkpoints, embeddings, exp_name):
        self.y_train = train_y
        self.y_test = test_y
        self.checkpoints = checkpoints
        self.enc = LabelEncoder()
        self.logreg = LogisticRegression
        self.exp_name = exp_name
   
    def load_data(self, path):
        with open(path, 'rb') as fin:
             data = pickle.load(fin)
        return np.asarray(data)

    def classify(self):
        """
        Trains a logistic regression and predicts labels
        :return: metrics of logistic regression perfomance
        """
        logreg = self.logreg()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        return (y_pred, accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='micro'),
              recall_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, 'micro')), logreg

    def write_to_files(self, pred):
        """
        Saves results to a file
        :param pred:
        :return:
        """
        pred = '\n'.join([','.join(list(map(str, i))) for i in pred])
        with open('pred.txt', 'a') as f:
            f.write(pred)
        scores = [[i, ] + list(a) for i, a in enumerate(scores)]
        sc = pd.DataFrame(scores, columns=['layer', 'accuracy', 'precision', 'recall', 'f1-score'])
        sc.to_csv('new_scores.csv', mode='a', header=False)

    def probe(self):
        predictions = []
        scores = []
        for train, test in self.checkpoints:
            pred = []
            score = []
            TRAIN = self.load_data(train)
            TEST = self.load_data(test)
            y_train = self.enc.fit_transform(self.y_train[i])
            y_test = self.enc.transform(self.y_test[i])
            for a in tqdm(range(7)):
                X_train = TRAIN[:, :, a]
                X_test = TEST[:, :, a]
                sc = self.classify(X_train, X_test, y_train, y_test)
                pred.append(sc[0])
                score.append(sc[1:5])
                with open(f'logreg_{self.exp_name}_{a}.pickle', 'wb') as fin:
                      pickle.dump(sc[-1], fin)
            print('Score is calculated')
            predictions.append(pred)
            scores.append(score)
        self.write_to_files(predictions, scores)
        return predictions, scores

