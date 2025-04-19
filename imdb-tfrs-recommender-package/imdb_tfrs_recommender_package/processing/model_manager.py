
import tensorflow as tf


# ==================== QUERY TOWER ARCHITECTURE =================== #
class UserModel(tf.keras.Model):
    """"
    Quey model architecture 

    args:
    * use_timestamp:bool
    * unique_user_ids: NDArray
    * timestamp: NDArray
    * timestamp_bucket: NDArray

    """

    def __init__(self, use_timestamp, unique_user_ids, timestamp, timestamp_bucket):
        super().__init__()
        
        self.use_timestamp =  use_timestamp
        self.unique_user_ids =  unique_user_ids
        self.timestamp =  timestamp
        self.timestamp_bucket = timestamp_bucket
        
        # converting user ids to integers and then to embeddings using keras preprocessing layers
        self.user_embedding = tf.keras.Sequential(
            [
            tf.keras.layers.StringLookup( # convert the string user ids to integer indices
                vocabulary = self.unique_user_ids, mask_token=None
            ),
            tf.keras.layers.Embedding( # convert the indices to vector embeddings
                len(self.unique_user_ids) + 1, 32
            )
            ]
        )

        #  incorporating timestamps to model user preferences at a point in time.
        
        # depending on the timestamp value it switches on and off this feature influence in our matrix computation 
        # dual operations: Firstly: obtain timestamp embeddings
        if self.use_timestamp:
            self.timestamp_embeddings = tf.keras.Sequential([
                tf.keras.layers.Discretization(
                    self.timestamp_bucket.tolist()
                ),
                tf.keras.layers.Embedding(
                    len(self.timestamp_bucket) + 1, 32
                )
            ])
            # Secondly normalize timestamp
            self.normalized_timestamp =  tf.keras.layers.Normalization(
                axis=None
            )
            self.normalized_timestamp.adapt(self.timestamp)

    def call(self, inputs):
        if not self.use_timestamp:
            return self.user_embedding(inputs['userID'])

        return tf.concat(
            [
            self.user_embedding(inputs['userID']),
            self.timestamp_embeddings(inputs['review date in unix']),
            tf.reshape(self.normalized_timestamp(inputs['review date in unix']), (-1, 1))
        ], axis=1)
        
# To capture more complex relationships, such as user preferences evolving over time, 
# we may need a deeper model with multiple stacked dense layers - Deep query retrieval model

# Full Query model
class QueryModel(tf.keras.Model):
    """ model for encoding user features """
    
    def __init__(self, layer_sizes, use_timestamp, unique_user_ids, timestamp, timestamp_bucket):

        """ initialize the user model embedding layer and the dense layer
        Args:
        * layer_sizes: a list of inttegers to create the dense layer depth

        * use_timestamp -  a boolean variable - that helps introduce some additional impact on the relationship between timing and user's preference
        
        """
        super().__init__()

        # pass the user model
        self.query_embedding_model = UserModel(use_timestamp, unique_user_ids, timestamp, timestamp_bucket)

        # add the dense layer
        self.dense_layers = tf.keras.Sequential()

        # using the relu activation for all the layers except the last. This helps introduce non-linearity for
        # studying complex relationships
        for layer in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer, activation='relu'))

        # for the last layer
        for layer in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer))

    def call(self, inputs):
        feature_embeddings = self.query_embedding_model(inputs)
        return self.dense_layers(feature_embeddings)
            
            
        
# ================ CANDIDATE TOWER ARCHITECTURE ================= #

class MovieModel(tf.keras.Model):
    """"
    Candidate model architecture 

    args:
    * unique_movie_titles: NDArray
    * unique_genres: NDArray

    """

    def __init__(self, unique_movie_titles, unique_genres):
        super().__init__()
        self.unique_movie_titles =  unique_movie_titles
        self.unique_genres =  unique_genres

        max_token = 10000 # maximum number of tokens to be generated in the vocabulary

        self.movie_embeddings = tf.keras.Sequential(
            [
            tf.keras.layers.StringLookup(
                vocabulary = self.unique_movie_titles, mask_token =None
            ),
            tf.keras.layers.Embedding(
                len(self.unique_movie_titles) + 1, 32
            )
            ]
        )

        self.genre_embeddings = tf.keras.Sequential([
            tf.keras.layers.TextVectorization(
                max_tokens=max_token,
                vocabulary= self.unique_genres
            ),
            tf.keras.layers.Embedding(
                max_token, 32, mask_zero=True
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

    def call(self, inputs):
        
        return tf.concat([
            self.movie_embeddings(inputs['originalTitle']),
            self.genre_embeddings(inputs['genres'])
        ], axis=1)

# Full candidate deep model

class CandidateModel(tf.keras.Model):

    """ model for encoding candidate features """

    def __init__(self, layer_sizes,unique_movie_titles, unique_genres):

        """ initialize the movie model embedding layer and the dense layer"""

        super().__init__()

        # pass the movie model
        self.candidate_embedding_model = MovieModel(unique_movie_titles, unique_genres)

        # add the dense layers
        self.dense_layers =  tf.keras.Sequential()

        for layer in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer, activation='relu'))

        # capturing the last dense layer
        for layer in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer))
                                  
    def call(self, inputs):

        feature_embeddings = self.candidate_embedding_model(inputs)
        return self.dense_layers(feature_embeddings)
