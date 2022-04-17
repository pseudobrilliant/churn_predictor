from tests.conftests import function_scope, default_data

import churn_library as churn
import logging
import os
import pytest

def test_import_data(function_scope):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		default_data_path = os.path.join(os.getenv("DATA_DIR"),"bank_data.csv")
		df = churn.import_data(default_data_path)
	except FileNotFoundError as err:
		logging.error("Testing import_data: The file wasn't found")
		raise err
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err

	logging.info("Testing import_data: SUCCESS")

def test_histplot_analysis(default_data):
	'''
	
	'''
	try:
		path = churn.histplot_analysis(default_data["df"], default_data["quantitative"][0], os.getenv("IMAGES_DIR"))
	except:
		logging
 
	assert path is not None
 