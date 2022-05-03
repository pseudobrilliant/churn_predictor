'''
Model library providing different model training methods
'''

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import pandas as pd

def train_linear_model(x_train:pd.DataFrame,
                       y_train:pd.DataFrame,
                       solver='lbfgs',
                       max_iter=3000):
    '''
    train, store model results: images + scores, and store models
    input:
        x_train: X training data
        y_train: Y training data
        solver: optional solver type for logistic regression
        max_iter: the max number of linear regression iterations

    output:
            None
    '''

    lrc = LogisticRegression(solver=solver, max_iter=max_iter)
    lrc.fit(x_train, y_train)

    return lrc


def train_random_forest_model_rfc(x_train: pd.DataFrame,
                                  y_train: pd.DataFrame,
                                  gridcv:int=5,
                                  param_grid:dict=None):
    '''
    train, store model results: images + scores, and store models
    input:
        x_train: X training data
        y_train: y training data
        cv: cv value used by grid search
        param_grid: parameters used in setting up the grid search
    output:
        None
    '''
    rfc = RandomForestClassifier(random_state=42)

    if param_grid is None:
        param_grid = {
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth' : [4,5,100],
                'criterion' :['gini', 'entropy']
        }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=gridcv)
    cv_rfc.fit(x_train, y_train)

    return cv_rfc.best_estimator_
