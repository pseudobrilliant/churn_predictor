'''
Data library containing helpful data management and processing functions
'''

import logging
import pandas as pd

from sklearn.model_selection import train_test_split

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
        logging.error('Unable to open file at path %s', path)
        raise
    except PermissionError:
        logging.error('Unable to open file at path %s due to permission issues', path)
        raise
    except pd.errors.ParserError:
        logging.error('Unable to parse file at path %s', path)
        raise

def category_mean_group_encoder(df, category_list:list, churn_col:str) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
            df: pandas dataframe
            category_list: list of columns that contain categorical features
            churn_col: column by which to group and calculate mean churn

    output:
            df: pandas dataframe with new calculated group means columns
    '''
    for category in category_list:
        name = f"{category}_{churn_col}"
        group_means = df.groupby(category).mean()[churn_col]
        values_list = [group_means.loc[val] for val in df[category]]
        df[name] = values_list

    return df

def data_filter_split(df,
                      target_col:str,
                      keep_col:list,
                      test_size:float=0.3,
                      rd_state:int=42):
    '''
    Filters and splits data frame into train-test splits

        input:
            target_col: target column representing our y data
            keep_col: list of columns to use as x features
            test_size: optional value used to set size of testing set
            rd_sate: random state used to seed random split

        output:
            x_train: dataframe containing training feature data
            x_test: dataframe containing testing feature data
            y_train: dataframe containing training target values
            y_test: dataframe containing testing target value

    '''
    try:
        x = df[keep_col]
        y = df[target_col]
    except KeyError as key_err:
        logging.error('Key value %s not found in dataframe', key_err)
        raise

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=rd_state)
    return  x_train, x_test, y_train, y_test
