# Customer Churn Prediction
Author: Chris Blythe
Course: Udacity Machine Learning DevOps

## Project Description
The goal of this project is to Modify and refactor an *existing* churn prediction
codebase to follow engineering and software best practices. The original code has
been modified into two sets of of code: reusable libraries for data analysis,
data managment, and model training and a churn predictor class with specific methods
to apply the libraries for churn specific data. Additional modifications include
the addition of a pytest testing suite of unit and integration tests and support
for running the main application through the Hydra configuration management library.

## Files and data description

  - churn_predictor.py: The main entry point for running churn analysis and
  model training. Can be run stand alone with the use of a hydra config file.

  - libraries: Contain generic data analysis, data management, and model training
  code that can be reused by other data science classes and experiments.

  - tests: Contains the unit tests and integration tests for the libraries
  and churn_predictor.

  - conf/config.yaml: Hydra configuration file used to identifie the target data
  file, target column containing attrition / churn, categorical columns,
  quantitative columns, and training columns to analyze and train models on.

  - data: Directory containing the data to analyze and train models on. This file
  must be created either manually or by the setup.sh script. Data provided here
  will then be referrenced by file name in the hydra configuration.

  - outputs: Contains processed data, data analysis reports, saved models, and
  model reports generated by the churn predictor. This directory is auto generated
  Hydra for each run of the predictor and saves the specific results as well as
  the provided configuration.


## How To Run

The below steps will guide you on how to setup the repo, run the stand alone
churn application, and run the tests included.

### Dependencies

In order to setup and run this repo you must have a current version of python
(3.8+) and virtualenv installed. All python librarydependencies for this repo are
detailed in the requirements.txt provided.

### Setup

For ease of use a simple setup.sh script has been provided. This script will
create all required folders and the python virtual environment with all dependencies.

To run the setup script run the following command within the repo directory.

`source setup.sh`

### Testing

Test are provided to check the unit and integration functionality of key library
and churn predictor functions.

To run the tests please use the pytest testing library as follows.

`pytest ./tests`

### Running

To run the main application 2 pre-requisites are required:
  - A config.yaml containing the expected churn condiguration must be provided
  within the `./conf` directory. An example of what the configuration file should
  contain can be found in `tests/test_conf`.

  - A `./data` directory containing any of the files you would like to analyze
  and train models on.

Once the above needs are met the main churn prediction application can be run as
follows:

`python churn_predictor.py`

## Continued Improvements

The following are some additional areas of improvement that could be made to this
solution:

  - Instead of hardcoded model training lists and analysis, make these optional
  parts of the hydra configuration. This way users can select which analysis and
  model training they would like to apply.

  - Allow for multiple hydra configurations so multiple data sets, analysis, and
  training can be run at once.
