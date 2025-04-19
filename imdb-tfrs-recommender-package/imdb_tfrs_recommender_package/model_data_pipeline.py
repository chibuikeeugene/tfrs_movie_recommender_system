import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
from imdb_tfrs_recommender_package.config import core as c
from imdb_tfrs_recommender_package.processing import data_manager as dm

from loguru import logger





def data_processing_pipeline():
    """ this method handles initial data processing before model is trained on it"""
    # set seed
    tf.random.set_seed(42)

    # loading the dataset
    tf_data = dm.load_and_preprocess_dataset(
        filename1= 'title.basics.csv',
        filename2= 'title.user-rating.csv'
        )
    logger.info('csv files loaded successfully and final tensor data created...')

    # extracting features for vocabulary building operation
    features = dm.get_unique_feature_list_or_dict_for_vocab_building(dataset= tf_data)

    # splitting the dataset
    train_test_val_data = dm.train_test_val_split(
        df=tf_data
    )

    # extracting only needed features from our tensor data
    train_ds, test_ds, val_ds = dm.extract_feature_from_each_dataset_split(
        dataset=train_test_val_data
    )

    # for performance and memory optimization
    logger.info('performing caching, batching and prefetching operation for train, test and val dataset')
    cached_train = train_ds.cache('imdb-tfrs-recommender-package/imdb_tfrs_recommender_package/cache_train/').batch(1000).prefetch(tf.data.AUTOTUNE)
    cached_test = test_ds.cache('imdb-tfrs-recommender-package/imdb_tfrs_recommender_package/cache_test/').batch(1000).prefetch(tf.data.AUTOTUNE)
    cached_val = val_ds.cache('imdb-tfrs-recommender-package/imdb_tfrs_recommender_package/cache_val/').batch(1000).prefetch(tf.data.AUTOTUNE)

    return cached_train, cached_val, features
