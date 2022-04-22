# library doc string


# import libraries
from pandas.errors import ParserError
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import joblib
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns; sns.set()

def import_data(path : str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at path
        
    input:
            path: a path to the csv
    output:
            df: pandas dataframe
    '''	
    
    try:
        df = pd.read_csv(path)
        return df

    except FileNotFoundError as err:
        logging.error('Unable to open file at path %s' % path)
        raise
    except PermissionError:
        logging.error('Unable to open file at path %s due to permission issues' % path)
        raise
    except ParserError:
        logging.error('Unable to parse file at path %s' % path)
        raise
        


    return None

def histplot_analysis(df:pd.DataFrame, column:str, dir_path:str) -> None:
    '''
    Generates and saves histogram plots for a single attributes within a DataFrame

    input:

            df: DataFrame containing data for analysis
            column: Idientifies which single attribute to generate an analysis of
            path: Identifies the directory path at which the file should be stored
    output:
            None

    '''

    hist_path = os.path.join(dir_path,'histplots')
    os.mkdir(hist_path)

    full_path = os.path.join(hist_path,f'{column}_histplot.png')
    try:

        sns.histplot(df[column], stat='density', kde=True)
        plt.savefig(full_path)
        return full_path

    except KeyError:
        logging.error('Unable to find columns %s in dataframe' % column)
        raise
    except IOError:
        logging.error('Unable to write plot to path %s' % full_path)
        raise
        
    return None

def category_count_analysis(df:pd.DataFrame, column:str, dir_path:str) -> None:
    '''
    Generates and saves histogram plots for a single attributes within a DataFrame

    input:

            df: DataFrame containing data for analysis
            column: Idientifies which single attribute to generate an analysis of
            path: Identifies the directory path at which the file should be stored
    output:
            None

    '''

    full_path = os.path.join(dir_path,'category_counts',f'{column}_category_count.png')
    try:

        df[column].value_counts('normalize').plot(kind='bar')
        plt.savefig(full_path)

    except KeyError:
        logging.error('Unable to find columns %s in dataframe' % column)
    except IOError:
        logging.error('Unable to write plot to path %s' % full_path)    


def perform_eda(df:pd.DataFrame, cat_col:list, quant_col:list) -> None:
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''    
    base_path = os.getenv("IMAGES_DIR")
    
    quant_path = os.path.join(base_path,'quantitative_reports')
    for col in quant_col:
        histplot_analysis(df, col, quant_path)

    cat_path = os.path.join(base_path,'categorical_reports')
    for col in cat_col:
        category_count_analysis(df, col, cat_path)


def encoder_helper(df, category_lst, target_col, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    postfix = response if response else str.capitalize(target_col)
    
    for category in category_lst:
        name = f"{category}_{postfix}"
        group_means = df.groupby(category).mean()[target_col]
        values_list = [group_means.loc[val] for val in df[category]]
        df[name] = values_list

    return df


def perform_feature_engineering(df, target_col, keep_col):
    '''
    input:
              df: pandas dataframe

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    try:
        x = df[keep_col]
        y = df[target_col]
    except KeyError as key_err:
        logging.error("Key value not found in dataframe" % key_err)
        raise

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=42)
    return  x_train, x_test, y_train, y_test



def classification_report(y_true, y_pred, name, dir_path):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values

    output:
             None
    '''
    class_report_dir = os.path.join(dir_path,'classifications')
    os.mkdir(class_report_dir)

    full_path = os.path.join(class_report_dir,f'{name}_classifications.png')

    class_report = classification_report(y_true, y_pred, output_dict=True)

    logging.info("classification_report: %s results" % name)
    logging.info("%s" % str(class_report))

    sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)
    plt.savefig(full_path)

    
def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    images_dir = os.getenv("IMAGES_DIR")
    models_dir = os.getenv("MODELS_DIR")
    roc_dir = os.path.join(images_dir,"roc")
    os.mkdir(roc_dir)

    model_list = [train_linear_model, train_random_forest_model]

    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    for train in model_list:
        model = train(x_train, x_test, y_train, y_test)

        name = train.__name__

        y_train_preds = model.predict(x_train)
        classification_report(y_train, y_train_preds, f"{name}_train", images_dir)

        y_test_preds = model.predict(x_test)
        classification_report(y_test, y_test_preds, f"{name}_test", images_dir)

        plot_roc_curve(model, x_test, y_test, ax=ax, alpha=0.8)
        
        joblib.dump(model, os.path.join(models_dir, f'{name}_model.pkl'))
       
    plt.savefig(os.path(roc_dir,"all_models_roc.png"))


def train_linear_model(x_train, y_train):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)

    return lrc


def train_random_forest_model(x_train, y_train):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    return cv_rfc.best_estimator_
