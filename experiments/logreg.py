import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pickle


class LogRegClassification(object):

    def __init__(self, train_y, test_y, checkpoints, exp_name):
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

    def classify(self, X_train, X_test, y_train, y_test):
        """
        Trains a logistic regression and predicts labels
        :return: metrics of logistic regression perfomance
        """
        logreg = self.logreg()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        return [y_pred, accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='micro'),
              recall_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'), logreg]

    def write_to_files(self, pred, scores):
        """
        Saves results to a file
        :param pred:
        :return:
        """
        pred = '\n'.join([','.join(list(map(str, i))) for i in pred])
        with open('pred.txt', 'a') as f:
            f.write(pred)
        sc = pd.DataFrame(scores, columns=['epoche', 'layer', 'accuracy', 'precision', 'recall', 'f1-score'])
        sc.to_csv(f'new_scores_{self.exp_name}.csv', mode='a', header=False)
        return sc

    def probe(self):
        predictions = []
        scores = []
        for num, (train, test) in enumerate(self.checkpoints):
            pred = []
            TRAIN = self.load_data(train)
            TEST = self.load_data(test)
            y_train = self.enc.fit_transform(self.y_train)
            y_test = self.enc.transform(self.y_test)
            for a in tqdm(range(7)):
                X_train = TRAIN[:, :, a]
                X_test = TEST[:, :, a]
                sc = self.classify(X_train, X_test, y_train, y_test)
                pred.append(sc[0])
                score = [num * 100000, a,] + sc[1:5]
                scores.append(score)
                with open(f'logreg_{self.exp_name}_{num}_{a}.pickle', 'wb') as fin:
                    pickle.dump(sc[-1], fin)
            print(f'{num}/{len(self.checkpoins} done')
            predictions.append(pred)
        sc = self.write_to_files(predictions, scores)
        return predictions, sc

