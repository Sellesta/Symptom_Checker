import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin

class NaiveBayesClassifier(ClassifierMixin):

    def __init__(self, alpha = 1):
        self.alpha = alpha

    def fit(self, X, y):
        ylabel = y.name
        data = pd.concat((X,y), axis=1)
        
        d = 2
        k = len(np.unique(y))
        self.smoothed_conditionals = {}
        self.smoothed_priors = {}
        
        for idx, grp in data.groupby(by='prognosis'):
            grp.drop(['prognosis'], axis=1, inplace=True)
            self.smoothed_conditionals[idx] = (np.sum(grp) + self.alpha)/(grp.shape[0] + self.alpha*d)
            self.smoothed_priors[idx] = (grp.shape[0] + self.alpha)/(data.shape[0] + self.alpha*k)
        
        self.smoothed_conditionals = pd.DataFrame(self.smoothed_conditionals)
        return self

    def predict_proba(self, X):
        results_proba = []
        classes = self.smoothed_conditionals.columns
        for i in range(X.shape[0]):
            posteriors = []
            for c_j in classes:
                x_i = X.iloc[i,:].values
                cp_list = self.smoothed_conditionals[c_j].values
                #posterior for each class = prior of that class * product of all conditionals
                posteriors.append(self.smoothed_priors[c_j] * np.prod((1-x_i) + np.power(-1, x_i+1)*cp_list))
            
            posteriors = np.array(posteriors)
            #normalize prediction posteriors for each datapoint
            posteriors /= np.sum(posteriors)
            results_proba.append({classes[i]:posteriors[i] for i in range(len(classes))})

        return pd.DataFrame(results_proba)

    def predict(self, X):
        res_proba = self.predict_proba(X)
        idxs = np.argmax(res_proba.values, axis=1)
        return [res_proba.columns[i] for i in idxs]

    def predict_from_proba(self, pred_proba_mat):
        idxs = np.argmax(pred_proba_mat.values, axis=1)
        return [pred_proba_mat.columns[i] for i in idxs]

    def prediction_confidence(self, pred_proba_mat):
        return 1 + np.sum(pred_proba_mat * np.log(pred_proba_mat), axis=1)/np.log(pred_proba_mat.shape[1])

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
