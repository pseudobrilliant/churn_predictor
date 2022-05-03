'''
Tests churn predictor functions
'''

import logging
import os
import pathlib

import matplotlib as mpl

from churn_predictor import ChurnPredictor, MODELS_TO_TRAIN
from libraries.data_library import import_data
from tests.conftests import config_scope, churn_scope

def test_initialization(config_scope):
    '''
    Tests the ability to initialize the churn prediction object
    '''
    out_path, data_path, cfg = config_scope

    churn = ChurnPredictor(out_path)
    try:
        assert os.path.exists(os.path.join(out_path,'churn_prediction.log'))
        assert os.path.exists(os.path.join(out_path,'images'))
        assert os.path.exists(os.path.join(out_path,'models'))
        assert os.path.exists(os.path.join(out_path,'processed'))
        assert os.path.exists(os.path.join(out_path,'images','bivariate_reports'))
        assert os.path.exists(os.path.join(out_path,'images','categorical_reports'))
        assert os.path.exists(os.path.join(out_path,'images','model_reports'))
        assert os.path.exists(os.path.join(out_path,'images','quantitative_reports'))
    except AssertionError:
        logging.error('test_initialization: Unable to initialize expected paths')
        raise

    logging.info('test_initialization: SUCCESS')


def test_analysis(churn_scope):
    '''
    Tests the ability to create churn analysis reports
    '''
    out_path, data_path, cfg, churn = churn_scope

    churn.analyze_data(data_path,
                       cfg.categorical_columns,
                       cfg.quantative_columns)

    data_name = pathlib.Path(data_path).stem
    images_path = os.path.join(out_path,'images')

    cat_path = os.path.join(images_path,'categorical_reports')
    for cat in cfg.categorical_columns:
        try:
            file_name = str.lower(f'{data_name}_{cat}_category_count.png')
            assert os.path.exists(os.path.join(cat_path,file_name))
        except AssertionError:
            logging.error('test_analysis: Unable to create category reports')
            raise

    quant_path = os.path.join(images_path,'quantitative_reports')
    for quant in cfg.quantative_columns:
        try:
            file_name = str.lower(f'{data_name}_{quant}_histplot.png')
            assert os.path.exists(os.path.join(quant_path,file_name))
        except AssertionError:
            logging.error('test_analysis: Unable to create quantitative reports')
            raise

    quant_path = os.path.join(images_path,'bivariate_reports')
    try:
        file_name = f'{data_name}_corr_heatmap.png'
        assert os.path.exists(os.path.join(quant_path,file_name))
    except AssertionError:
        logging.error('test_analysis: Unable to create quantitative reports')
        raise

    logging.info('test_analysis: SUCCESS')

def test_processing(churn_scope):
    '''
    Tests the ability to process data and generate an accurate dataframe and file
    '''

    out_path, data_path, cfg, churn = churn_scope

    churn._process_data(data_path, cfg.target_column, cfg.categorical_columns)

    data_name = pathlib.Path(data_path).stem
    processed_path = os.path.join(out_path,'processed')

    file_name = f'{data_name}_processed.csv'
    file_path = os.path.join(processed_path,file_name)
    try:
        assert os.path.exists(file_path)
    except AssertionError:
        logging.error('test_processing: Unable to create processed data set')
        raise

    df = import_data(file_path)

    for cat in cfg.categorical_columns:
        try:
            assert f'{cat}_Churn' in df
        except AssertionError:
            logging.error('test_processing: processed data does not contain churn')
            raise

    logging.info('test_processing: SUCCESS')

def test_train_models(churn_scope):
    '''
    Tests the ability to train models based on the expected data and configuration
    '''
    out_path, data_path, cfg, churn = churn_scope

    mpl.use('Agg')

    churn.train_models(data_path,
                       cfg.target_column,
                       cfg.training_columns,
                       cfg.categorical_columns)

    data_name = pathlib.Path(data_path).stem
    images_path = os.path.join(out_path,'images')
    processed_path = os.path.join(out_path,'processed')

    model_rep_path = os.path.join(images_path,'model_reports')
    models_path = os.path.join(out_path,'models')

    try:
        file_name = f'{data_name}_processed.csv'
        assert os.path.exists(os.path.join(processed_path,file_name))
    except AssertionError:
        logging.error('test_train_models: Unable to create processed data set')
        raise

    for model in ['LogisticRegression', 'RandomForestClassifier']:
        try:
            lmodel = str.lower(model)
            file_name = f'{data_name}_{lmodel}_model.pkl'
            assert os.path.exists(os.path.join(models_path,file_name))
        except AssertionError:
            logging.error('test_train_models: Unable to create saved model files')
            raise

        try:
            file_name = f'{data_name}_{lmodel}_train_classifications.png'
            assert os.path.exists(os.path.join(model_rep_path,file_name))
            file_name = f'{data_name}_{lmodel}_test_classifications.png'
            assert os.path.exists(os.path.join(model_rep_path,file_name))
        except AssertionError:
            logging.error('test_train_models: Unable to create classification reports')
            raise

    try:
        assert os.path.exists(os.path.join(model_rep_path,'all_models_roc.png'))
    except AssertionError:
        logging.error('test_train_models: Unable to create roc reports')
        raise

    logging.info('test_train_models: SUCCESS')
