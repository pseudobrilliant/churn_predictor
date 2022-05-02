'''
Tests library functions
'''
import logging
import pytest

from libraries import data_library
from tests.conftests import config_scope, churn_scope

def test_import_data(config_scope):
    '''
    Tests the import_data function can consume a static file and
    generate an appropriate data frame
    '''
    try:
        df = data_library.import_data(config_scope[1])
    except FileNotFoundError:
        logging.error('test_import_data: Testing data file not found')
        raise
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError:
        logging.error('test_import_data: Data does not contain rows and columns')
        raise

    logging.info('test_import_data: SUCCESS')

def test_category_mean_group_encoder(config_scope):
    out_path, data_path, cfg = config_scope
    
    df = data_library.import_data(data_path)
    
    df['Churn'] = df[cfg.target_column].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    
    df = data_library.category_mean_group_encoder(df,
                                                  cfg.categorical_columns,
                                                  'Churn')
    
    for cat in cfg.categorical_columns:
        try:
            assert f'{cat}_Churn' in df
        except AssertionError:
            logging.error('test_category_mean_group_encoder: '
                          'processed data does not contain churn')
            raise