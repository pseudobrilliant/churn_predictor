'''
Initialize configurations for testing
'''

# Import standard libraries
import logging
import os
import pytest
import tempfile
import shutil

# Import third party libraries
from omegaconf import OmegaConf

# Import custom libraries
from churn_predictor import ChurnPredictor

@pytest.fixture(scope='session')
def config_scope():
    '''
    Provides a basic configuration fixture for testing
    '''
    cfg_path = './tests/test_conf/config.yaml'
    cfg = OmegaConf.load(cfg_path)

    out_path = './tests/outputs'
    os.makedirs(out_path, exist_ok=True)

    data_path = os.path.join('./tests/test_data/',cfg.data_file)

    logging.basicConfig(
    filename=os.path.join(out_path,'churn_prediction.log'),
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

    yield out_path, data_path, cfg

@pytest.fixture(scope='session')
def churn_scope(config_scope):
    '''
    Provides a churn predictor object as a fixture along with the basic
    configuration
    '''
    out_path, data_path, cfg = config_scope

    churn = ChurnPredictor(out_path)

    yield out_path, data_path, cfg, churn