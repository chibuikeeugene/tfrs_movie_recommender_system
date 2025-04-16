import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from imdb_tfrs_recommender_package.config import core as c
from imdb_tfrs_recommender_package.processing import data_manager as dm

import pandas as pd
from loguru import logger





def run_training():
    """ create new model"""

    # loading the dataset
    tf_data = dm.load_and_preprocess_dataset(
        filename1= f'{c.DATASET}/title.basics.csv',
        filename2=f'{c.DATASET}/title.user-rating.csv'
        )
    logger.info('csv files loaded successfully and final tensor data created...')

    # extracting features
    features = dm.get_unique_feature_list_or_dict_for_vocab_building(dataset= tf_data)

    # splitting the dataset
    train_test_val_data = dm.train_test_val_split(
        df=tf_data
    )

    # extracting only needed features from our tensor data
    train_ds, test_ds, val_ds = dm.extract_feature_from_each_dataset_split(
        dataset=train_test_val_data
    )

    

if __name__ == '__main__':
    run_training()