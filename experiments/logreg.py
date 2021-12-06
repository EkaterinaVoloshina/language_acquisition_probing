import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


class LogRegClassification(object):

    def __init__(self, objects, labels, checkpoints, embeddings):
        self.X_train, self.X_test = objects
        self.y_train, self.y_test = labels
        self.checkpoints = checkpoints
        self.enc = LabelEncoder()
        self.logreg = LogisticRegression()

    def classify(self):
        """
        Trains a logistic regression and predicts labels
        :return: metrics of logistic regression perfomance
        """
        self.logreg.fit(X_train, y_train)
        y_pred = self.logreg.predict(X_test)
        return (y_pred, accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='micro'),
              recall_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, 'micro'))

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
        for i in self.checkpoints:
            pred = []
            score = []
            TRAIN = self.X_train[i]
            TEST = self.X_test[i]
            y_train = self.enc.fit_transform(self.y_train[i])
            y_test = self.enc.transform(self.y_test[i])
            for a in tqdm(range(7)):
                train = np.array([x[a].tolist() for x in TRAIN.to_list()])
                test = np.array([x[a].tolist() for x in TEST.to_list()])
                sc = self.classify(train, test, y_train, y_test)
                pred.append(sc[0])
                score.append(sc[1:])
            print('Score is calculated')
            predictions.append(pred)
            scores.append(score)
        self.write_to_files(predictions, scores)
        with open('tfidf.pickle', 'wb') as fin:
            pickle.dump(self.logreg, fin)
        return predictions, scores

