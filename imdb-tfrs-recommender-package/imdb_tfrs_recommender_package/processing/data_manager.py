from pathlib import Path
import datetime
from imdb_tfrs_recommender_package.config.core import DATASET, TRAINED_MODEL
import numpy as np
from loguru import logger
import pandas as pd
import tensorflow as tf
import keras


def load_and_preprocess_dataset(*, filename1:str, filename2:str) -> tf.data.Dataset:
    """ load the data from the mysql database or data warehouse, 
    or just the csv files from a dataset directory. 
    function block can be expanded by providing connection parameters.
    Converts the datframe to a tensor dataset

    args:
    * filename1:str - ratings tabular data or csv file
    * filename2:str = users tabular data or csv file

    returns:
    * a tensor data
    """

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
    logger.info('wrapping things up with the final dataframe by removing review dates and converting it to a tensor datatset...')
    df = df.drop('review date', axis=1)

    df_tensor = tf.data.Dataset.from_tensor_slices(df.to_dict('list'))
    

    return df_tensor

def get_unique_feature_list_or_dict_for_vocab_building(dataset):
    """ call this function to obtain unique features for vocab needed in training

    arg:
    * dataset - a tensor data Dataset

    return:
    * a dictionary of feature items:

    * Keys:
    'movie_titles'
    'genres'
    'movietitle_genres'
    'timestamp'
    'user_ids'
    
    """
    
    df_tensor = dataset

    # movie titles 
    movie_titles =  df_tensor.map(lambda x: x['originalTitle'])
    movie_titles_batched = movie_titles.batch(1000)
    unique_movie_titles = np.unique(np.concatenate(list(movie_titles_batched)))

    # select just the genre
    genres =  df_tensor.map(lambda x: x['genres'])
    genres_batched = genres.batch(1000)
    unique_genres = np.unique(np.concatenate(list(genres_batched)))

    movietitle_genres = df_tensor.map( lambda x: {
            'originalTitle' : x['originalTitle'],
            'genres': x['genres']
        }
    )

    # select just the review date unix timestamp
    timestamp =  df_tensor.map(lambda x: x['review date in unix'])
    timestamp_batched=timestamp.batch(1000)
    timestamp =  np.concatenate(list(timestamp_batched))
    max_timestamp = timestamp.max()
    min_timestamp = timestamp.min()

    timestamp_bucket = np.linspace(
        min_timestamp, max_timestamp, num=1000)

    # selecting just the user id
    user_ids = df_tensor.map(lambda x: x['userID'])
    user_ids_batched = user_ids.batch(1000)
    unique_user_ids = np.unique(np.concatenate(list(user_ids_batched)))

    logger.info('returning unique feature list/dict items for movie titles, genres, timestamp, uniqe_ids...')

    return {'movie_titles': unique_movie_titles,
            'genres': unique_genres,
            'movietitle_genres':movietitle_genres,
            'timestamp':timestamp_bucket,
            'user_ids':unique_user_ids 
            }


def train_test_val_split(df: tf.data.Dataset) -> list[tf.data.Dataset]:
    """ split the dataframe into train , test and val tf dataset"""

    df_tensor_len = len(df)

    train_size = int(0.7 * df_tensor_len)
    val_size = int(0.15 * df_tensor_len)
    # test_size = df_tensor_len - train_size - val_size

    train_ds = df.take(train_size)
    val_ds = df.skip(train_size).take(val_size)
    test_ds = df.skip(train_size + val_size)
    logger.info(f'{len(df)} =  {len(train_ds)} + {len(test_ds)} + {len(val_ds)}')
    return [train_ds, test_ds, val_ds]

def extract_feature_from_each_dataset_split(dataset:list[tf.data.Dataset]):
    """ extract features from each dataset split and create new json objects for training """
    train = dataset[0]
    test = dataset[1]
    val = dataset[2]

    train_ds = train.map(lambda x : {
        'userID': x['userID'],
        'originalTitle': x['originalTitle'],
        'rating':x['rating'],
        'genres': x['genres'],
        'runtimeMinutes': x['runtimeMinutes'],
        'review date in unix': x['review date in unix']
    })
    test_ds = test.map(lambda x : {
        'userID': x['userID'],
        'originalTitle': x['originalTitle'],
        'rating':x['rating'],
        'genres': x['genres'],
        'runtimeMinutes': x['runtimeMinutes'],
        'review date in unix': x['review date in unix']
    })
    val_ds = val.map(lambda x : {
        'userID': x['userID'],
        'originalTitle': x['originalTitle'],
        'rating':x['rating'],
        'genres': x['genres'],
        'runtimeMinutes': x['runtimeMinutes'],
        'review date in unix': x['review date in unix']
    })

    return train_ds, test_ds, val_ds