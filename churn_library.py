# library doc string


# import libraries
from pandas.errors import ParserError

import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns; sns.set()

os.environ['QT_QPA_PLATFORM']='offscreen'



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

    except FileNotFoundError:
        logging.error('Unable to open file at path %s' % path)
    except PermissionError:
        logging.error('Unable to open file at path %s due to permission issues' % path)
    except ParserError:
        logging.error('Unable to parse file at path %s' % path)


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

    full_path = os.path.join(dir_path,'histplots',f'{column}_histplot.png')
    try:

        sns.histplot(df[column], stat='density', kde=True)
        plt.savefig(full_path)

    except KeyError:
        logging.error('Unable to find columns %s in dataframe' % column)
    except IOError:
        logging.error('Unable to write plot to path %s' % full_path)

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


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
    ]

    quant_columns = [
        'Customer_Age',
        'Dependent_count', 
        'Months_on_book',
        'Total_Relationship_Count', 
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 
        'Credit_Limit', 
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 
        'Total_Amt_Chng_Q4_Q1', 
        'Total_Trans_Amt',
        'Total_Trans_Ct', 
        'Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio'
    ]
    
    base_path = 'images'
    
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