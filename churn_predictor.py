'''
Churn Predictor class used to analyze and train models on churn related data.
'''

# Standard Library imports
import logging
import os
import pathlib

# Third party library imports
from omegaconf import DictConfig
from sklearn.metrics import plot_roc_curve

import hydra
import joblib
import matplotlib.pyplot as plt
import pandas as pd

#Internal library imports
from libraries import analysis_library, data_library, model_library

# Constant analysis settings and model training for Churn Prediction
QUANT_ANALYSIS = [analysis_library.histplot_analysis]
CAT_ANALYSIS = [analysis_library.category_count_analysis]
BIVARIATE_ANALYSIS = [analysis_library.category_count_analysis]
MODELS_TO_TRAIN = [
                    model_library.train_linear_model,
                    model_library.train_random_forest_model_rfc
                ]
CUSTOM_ANALYSIS = [
                    None,
                    [analysis_library.feature_class_analysis,
                        analysis_library.feature_importance_analysis]
                ]

class ChurnPredictor():
    '''
    Churn Predictor class used to analyze and train models on churn related data.
    '''

    def __init__(self, out_dir:str) -> None:
        '''
        Churn Predictor conrstructor to allow for analysis and model training
        on provided churn data.
            input:
                out_dir: Path of target output directory for all content
                generated by the churn predictor class.
        '''

        self._initialize_paths(out_dir)

    def _initialize_paths(self, out_dir:str) -> None:
        '''
        Initialize data, images, and models path within the output directory
        path provided to the constructor.
            input:
                out_dir: Path of target output directory for all content
                generated by the churn predictor class.

            output:
                None
        '''

        if not os.path.exists(out_dir):
            logging.error('Output directory %s does not exist. Unable to'
                'initialize ChurnPredictor', out_dir)
            raise FileNotFoundError

        self.out_dir = out_dir

        logging.info('Initializing paths at output directory %s', self.out_dir)

        try:
            # Initializing core directories
            self.processed_dir = os.path.join(self.out_dir, 'processed')
            os.makedirs(self.processed_dir,exist_ok=True)

            self.images_dir = os.path.join(self.out_dir, 'images')
            os.makedirs(self.images_dir,exist_ok=True)

            self.models_dir = os.path.join(self.out_dir, 'models')
            os.makedirs(self.models_dir,exist_ok=True)

            # Initializing report directories under images
            report_dirs = [
                            'quantitative_reports',
                            'categorical_reports',
                            'bivariate_reports',
                            'model_reports'
                        ]

            self.report_dirs = {}

            for report in report_dirs:
                report_path = os.path.join(self.images_dir, report)
                os.makedirs(report_path, exist_ok=True)
                self.report_dirs[report] = report_path

        except PermissionError:
            logging.error('Insufficient permissions on output directory %s'
            'Unable to initialize Churn Predictor', self.out_dir)
            raise


    def analyze_data(self, data_path:str, cat_col:list, quant_col:list) -> None:
        '''
        perform exploratory data analysis on given data and save figure reports
        to images folder
            input:
                data_path: path to input files targeted for churn analysis
                cat_col: a list of categorical columns within the target data
                quant_col: a list of quantifiable columns within the target data

            output:
                None
        '''

        logging.info('Starting exploraroty data analysis')

        # Reading data into datframe
        df = data_library.import_data(data_path)
        if df is None:
            logging.info('Unable to import data and complete analysis')
            return

        data_name = pathlib.Path(data_path).stem

        logging.info('Generating univariate reports for quantitative columns')
        for col in quant_col:
            for analysis in QUANT_ANALYSIS:
                analysis(df,
                         col,
                         self.report_dirs['quantitative_reports'],
                         data_name)

        logging.info('Generating univariate reports for categorical columns')
        for col in cat_col:
            for analysis in CAT_ANALYSIS:
                analysis(df,
                         col,
                         self.report_dirs['categorical_reports'],
                         data_name)

        logging.info('Generating reports for multiple variables')
        analysis_library.heatmap_analysis(df,
                                          self.report_dirs['bivariate_reports'],
                                          data_name)

        logging.info('Completed exploratory data analysis')


    def train_models(self,
                     data_path:str,
                     target_col:str,
                     keep_col:list,
                     cat_col:list):
        '''
        train, store model results: images + scores, and store models
            input:
                data_path: path to input files targeted for training
                target_col: identifies which column will be used to calculate churn
                keep_col: list of columns to maintain in data during training
                cat_col: a list of categorical columns within the target data

            output:
                model_result_list: a list of all the models generated
        '''

        df = self._process_data(data_path, target_col, cat_col)

        if df is None:
            logging.info('Unable to proccess data and complete training')
            return None

        data_name = pathlib.Path(data_path).stem

        x_train, x_test, y_train, y_test = data_library.data_filter_split(df,
                                                                   target_col,
                                                                   keep_col)

        logging.info('Beginning model training')
        result_model_list = []
        for model_training in MODELS_TO_TRAIN:

            logging.info("Running training for %s", model_training.__name__)

            model = model_training(x_train, y_train)
            result_model_list.append(model)

            name = str.lower(type(model).__name__)

            mpath = os.path.join(self.models_dir, f'{data_name}_{name}_model.pkl')

            logging.info('Saving a %s model to %s', name, mpath)

            joblib.dump(model, mpath)

        logging.info('Completed model training')

        self._analyze_models(x_train,
                             x_test,
                             y_train,
                             y_test,
                             result_model_list,
                             data_name)

        return result_model_list

    def _process_data(self,
                      data_path:str,
                      target_col:str,
                      category_col:list) -> pd.DataFrame:
        '''
        Processes data by calculating churn and encoding categorical columns
            input:
                data_path: path to input files targeted for processing
                target_col: identifies which column represents the target churn
                keep_col: list of columns to maintain in data during training
                cat_col: a list of categorical columns within the target data

            output:
                df: processed dataframe containing churn and encoded values
        '''

        logging.info('Processing data from path %s', data_path)
        df = data_library.import_data(data_path)

        if df is None:
            logging.error('Unable to import data and complete processing')
            return None

        data_name = pathlib.Path(data_path).stem

        logging.info('Calculating churn based on column %s', target_col)

        try:
            df['Churn'] = df[target_col].apply(
                lambda val: 0 if val == 'Existing Customer' else 1)
        except KeyError:
            logging.error('Target churn column %s does not exist', target_col)

        df = data_library.category_mean_group_encoder(df,
                                                      category_col,
                                                      'Churn')

        if df is None:
            logging.error('Unable to encode data and complete processing')
            return None

        out_path = os.path.join(self.processed_dir, f'{data_name}_processed.csv')
        logging.info('Saving processed data to %s', out_path)

        df.to_csv(out_path) # pylint: disable=no-member

        return df


    def _analyze_models(self,
                        x_train:pd.DataFrame,
                        x_test:pd.DataFrame,
                        y_train:pd.DataFrame,
                        y_test:pd.DataFrame,
                        models:list,
                        prefix:str=None):
        '''
        Analyzes models to produce classification, roc, and custom reports per model
            input:
                x_train: dataframe containing training feature data
                x_test: dataframe containing testing feature data
                y_train: dataframe containing training target values
                y_test: dataframe containing testing target value
                models: list of trained models to analyze
                prefix: optional values used to add prefix to output
            output:
                None
        '''

        logging.info('Analyzing models')
        rep_path = self.report_dirs['model_reports']
        names = []
        plt.clf()
        for model in models:
            name = str.lower(type(model).__name__ if prefix is None else
                f'{prefix}_{type(model).__name__}')
            names.append(name)
            logging.info('Analyzing %s classification performance', name)

            y_train_preds = model.predict(x_train)
            analysis_library.classification_report_analysis(y_train,
                                                            y_train_preds,
                                                            f'{name}_train',
                                                            rep_path)

            y_test_preds = model.predict(x_test)
            analysis_library.classification_report_analysis(y_test,
                                                            y_test_preds,
                                                            f'{name}_test',
                                                            rep_path)


        logging.info('Analyzing model ROC performance')

        plt.clf()
        plt.figure(figsize=(15, 8))
        fig_plot = plt.gca()
        for model in models:
            plot_roc_curve(model, x_test, y_test, ax=fig_plot, alpha=0.8)
        plt.savefig(os.path.join(self.report_dirs['model_reports'],
                                'all_models_roc.png'))
        plt.close()

        plt.clf()
        for i,model in enumerate(models):
            if CUSTOM_ANALYSIS[i] is None:
                continue
            for analysis in CUSTOM_ANALYSIS[i]:
                logging.info('Running custom analysis for %s'
                             '(This may take 10 - 15 minutes)', names[i])

                analysis(model,
                        x_test,
                        f'{name}_test',
                        self.report_dirs['model_reports'])


@hydra.main(config_path='conf', config_name='config')
def run_predictor(cfg : DictConfig) -> None:
    '''
        Uses a Churn Predictor class to analyze data, train models, and analyze
        the results. Provides values to churn predictor through the use of a
        hydra configuration.
    '''

    logging.info('Running main cli function using hydra configuration')

    # Hydra will change directory to expected output directory this allows us
    # to pass the output directory to the churn predictor
    out_dir = os.getcwd()

    data_path = os.path.join(hydra.utils.get_original_cwd(),'data',cfg.data_file)

    churn_predictor = ChurnPredictor(out_dir)

    churn_predictor.analyze_data(data_path,
                                 cfg.categorical_columns,
                                 cfg.quantative_columns)

    churn_predictor.train_models(data_path,
                                 cfg.target_column,
                                 cfg.training_columns,
                                 cfg.categorical_columns)

    logging.info('Completed cli run of predictor')

if __name__ == '__main__':
    run_predictor() #pylint: disable=no-value-for-parameter
