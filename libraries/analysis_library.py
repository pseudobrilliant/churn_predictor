'''
Data analysis module containing functions to analyze univariate, multivariate,
bivariate data and various types of models.
'''

# Importing standard libraries
import logging
import os

# Importing third party libraries
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

def histplot_analysis(df:pd.DataFrame, column:str, dir_path:str, prefix:str=None) -> None:
    '''
    Generates and saves histogram plots for a single attributes within a DataFrame

    input:
            df: DataFrame containing data for analysis
            column: Idientifies which single attribute to generate an analysis of
            dir_path: Identifies the directory path at which the file should be stored
            prefix: optional prefix to add to saved reports
    output:
            None

    '''

    name = str.lower(f'{prefix}_{column}' if prefix else column)

    full_path = os.path.join(dir_path,f'{name}_histplot.png')
    try:

        plt.clf()
        sns.histplot(df[column], stat='density', kde=True)
        plt.savefig(full_path)
        plt.close()

        logging.info('Saved histplots for %s in %s', column, full_path)

    except KeyError:
        logging.error('Unable to find columns %s in dataframe', column)
        raise
    except IOError:
        logging.error('Unable to write plot to path %s', full_path)
        raise

def category_count_analysis(df:pd.DataFrame, column:str, dir_path:str, prefix:str=None) -> None:
    '''
    Generates and saves histogram plots for a single attributes within a DataFrame

    input:
            df: DataFrame containing data for analysis
            column: Idientifies which single attribute to generate an analysis of
            dir_path: Identifies the directory path at which the file should be stored
            prefix: optional prefix to add to saved reports
    output:
            None

    '''

    name = str.lower(f'{prefix}_{str.lower(column)}' if prefix else column)

    full_path = os.path.join(dir_path,f'{name}_category_count.png')
    try:

        plt.clf()
        df[column].value_counts('normalize').plot(kind='bar')
        plt.savefig(full_path)
        plt.close()

        logging.info('Saved value counts for %s in %s', column, full_path)


    except KeyError:
        logging.error('Unable to find columns %s in dataframe', column)
    except IOError:
        logging.error('Unable to write plot to path %s', full_path)


def heatmap_analysis(df:pd.DataFrame, dir_path:str, prefix=None) -> None:
    '''
    Generates and saves histogram plots for a single attributes within a DataFrame

    input:

            df: DataFrame containing data for analysis
            column: Idientifies which single attribute to generate an analysis of
            path: Identifies the directory path at which the file should be stored
    output:
            None

    '''

    name = str.lower(prefix if prefix else "")

    full_path = os.path.join(dir_path,f'{name}_corr_heatmap.png')
    try:

        plt.clf()
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig(full_path)
        plt.close()

        logging.info('Saving heatmap to %s', full_path)

    except IOError:
        logging.error('Unable to write plot to path %s', full_path)


def classification_report_analysis(y_true, y_pred, name, dir_path, prefix=None):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values

    output:
        None
    '''

    name = str.lower(f'{prefix}_{name}' if prefix else name)

    full_path = os.path.join(dir_path,str.lower(f'{name}_classifications.png'))

    class_report = classification_report(y_true, y_pred, output_dict=True)

    logging.info("classification_report: %s results", name)
    logging.info("%s", str(class_report))

    plt.clf()
    sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)
    plt.savefig(full_path)
    plt.close()


def feature_class_analysis(model,
                           x_vals:pd.DataFrame,
                           name:str,
                           dir_path:str,
                           prefix:str=None):
    '''
    Plots the result of features on classification

        input:
            model: The model by which to analze features over cassification
            x_vals: values over which to form analysis
            name: name of model
            dir_path: where to save report
            prefix: optional prefix to add to saved file
        output:
            None
    '''

    name = str.lower(f'{prefix}_{name}' if prefix else name)
    full_path = os.path.join(dir_path,str.lower(f'{name}_feature_summary.png'))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_vals)

    plt.clf()
    shap.summary_plot(shap_values, x_vals, plot_type="bar")
    plt.savefig(full_path)
    plt.close()

def feature_importance_analysis(model,
                                x_vals:pd.DataFrame,
                                name:str,
                                dir_path:str,
                                prefix:str=None):
    '''
    Plots the result of features on classification

        input:
            model: The model by which to analze feature immportance
            x_vals: values over which to form the analysis
            name: name of model
            dir_path: where to save report
            prefix: optional prefix to add to saved file
        output:
            None
    '''

    full_name = str.lower(f'{prefix}_{name}' if prefix else name)
    full_path = os.path.join(dir_path,str.lower(f'{full_name}_feature_importance.png'))

    if not hasattr(model, "feature_importances_"):
        logging.warning("Feature importance not applicable to model %s", name)
        return

    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_vals.columns[i] for i in indices]

    plt.clf()

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_vals.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_vals.shape[1]), names, rotation=90)

    plt.savefig(full_path)
    plt.close()
