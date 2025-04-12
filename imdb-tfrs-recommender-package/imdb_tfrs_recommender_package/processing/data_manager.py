from pathlib import Path
import datetime
from imdb_tfrs_recommender_package.config.core import DATASET, TRAINED_MODEL
from loguru import logger
import pandas as pd
import tensorflow as tf
import keras


def load_and_preprocess_dataset(*, filename1:str, filename2:str) -> pd.DataFrame:
    """ load the data from the mysql database or data warehouse, or just the csv files from a dataset directory. 
    function block can be expanded by providing connection parameters"""

    ratings_data = pd.read_csv(f'{DATASET}/{filename1}')
    users_data = pd.read_csv(f'{DATASET}/{filename2}')

    #select needed features from ratings
    ratings_data = ratings_data[['tconst', 'originalTitle', 'genres', 'runtimeMinutes']]

    # rename column
    ratings_data.rename(columns={'tconst': 'movieID'}, inplace=True)

    data = users_data.merge(ratings_data, on='movieID')

    logger.info(f'shape of the data: {data.shape}')

    # create a new movie title list that enforces type uniformity in its values
    updated_movie_titles = []

    for i in data['originalTitle']:
        if isinstance(i, str):
            updated_movie_titles.append(i)
        else:
            a = str(i)
            updated_movie_titles.append(a)
    
    data['originalTitle'] = updated_movie_titles

    # Due to data size and training time, we will use some records
    df = data.sample(n = 100000, random_state = 50, axis = 0)

    # convert the review date to datetime
    df['review date'] = pd.to_datetime(df['review date'])

    # convert the datetime to just unix timestamp
    df['review date in unix'] = [datetime.datetime.timestamp(time) for time in df['review date']]

    # sort the data by review date so that we can split the data into training, testing and validation sets
    logger.info('sorting dataframe by review date in ascending order...')
    df.sort_values(by='review date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # drop the review date column
    df = df.drop('review date', axis=1)

    return df

def train_test_val_split(df: pd.DataFrame) -> list[pd.DataFrame]:
    """ split the dataframe into train , test and val data"""

    train = df[:int(0.7*len(df))]
    test = df[int(0.7*len(df)):int(0.85*len(df))]
    val = df[int(0.85*len(df)):]
    logger.info(f"train data shape: {train.shape}")
    logger.info(f'test data shape: {test.shape}')
    logger.info(f'val data shape: {val.shape}')
    return [train, test, val]

def convert_dataframe_to_tensors(data: list[pd.DataFrame]):
    """" convert the train, test and val into a tensorflow dataset """
    train = data[0]
    test =  data[1]
    val = data[2]
    train_df = tf.data.Dataset.from_tensor_slices(train.to_dict('list'))
    test_df = tf.data.Dataset.from_tensor_slices(test.to_dict('list'))
    val_df = tf.data.Dataset.from_tensor_slices(val.to_dict('list'))
