# library doc string


# import libraries
from pandas.errors import ParserError

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
    for col in quant_columns:
        histplot_analysis(df, col, quant_path)

    cat_path = os.path.join(base_path,'categorical_reports')
    for col in cat_columns:
        category_count_analysis(df, col, cat_path)

def encoder_helper(df, category_lst, response):
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
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


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

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass