import logging
import os
import pytest
import tempfile
import shutil

from churn_library import import_data
from dotenv import load_dotenv

dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)

@pytest.fixture(scope="session")
def function_scope():

    clear_dirs = [os.getenv("DATA_DIR"), 
                  os.getenv("LOGS_DIR"), 
                  os.getenv("IMAGES_DIR"), 
                  os.getenv("MODELS_DIR")]
    for path in clear_dirs:
        shutil.rmtree(path,ignore_errors=True)
        os.makedirs(path,exist_ok=False)
    
    src_path = os.path.join(os.getcwd(),"tests","test_data","bank_data.csv")
    dst_path = os.path.join(os.getenv("DATA_DIR"),"bank_data.csv")
    shutil.copyfile(src_path, dst_path)
    
    logging.basicConfig(
    filename=os.path.join(os.getenv("LOGS_DIR"),"testing_churn_prediction.log"),
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
    
    yield
    
@pytest.fixture
def default_data(function_scope):
    cat_cols = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]

    quant_cols = [
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
    
    default_data_path = os.path.join(os.getenv("DATA_DIR"), "bank_data.csv")
    yield {"df": import_data(default_data_path), "categorical": cat_cols, "quantitative": quant_cols}