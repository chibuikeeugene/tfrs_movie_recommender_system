import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
import tensorflow_recommenders as tfrs
from  imdb_tfrs_recommender_package.processing import model_manager
from imdb_tfrs_recommender_package import model_data_pipeline
from loguru import logger

# using the tfrs.Model class to wrap our two-tower model architecture and define metrics and loss functions

class FinalModel(tfrs.models.Model):
    def __init__(self,
                 layer_sizes,
                 use_timestamp,
                 movietitle_genres,
                 unique_user_ids,
                 timestamp,
                 timestamp_bucket,
                 unique_movie_titles,
                 unique_genres
                 ):
        super().__init__()
    
        self.query_model: tf.keras.Model = model_manager.QueryModel(layer_sizes,
                                                                    use_timestamp,
                                                                    unique_user_ids,
                                                                    timestamp,
                                                                    timestamp_bucket
                                                                    )
        self.candidate_model: tf.keras.Model = model_manager.CandidateModel(layer_sizes,
                                                                            unique_movie_titles,
                                                                            unique_genres
                                                                            )
        self.tasks =  tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates= movietitle_genres.batch(128).map(self.candidate_model)
            )
        )

    
    def compute_loss(self, features, training = False) -> tf.Tensor:

        # pass the user id feature
        user_embeddings = self.query_model(features)

        # pass the movie title feature
        positive_movie_embeddings = self.candidate_model(features)

        metrics_and_loss = self.tasks(user_embeddings, positive_movie_embeddings)

        return metrics_and_loss


#=============== run model training ==============#

def run_model_training():

    # load the cached dataset
    cached_train, cached_val, features = model_data_pipeline.data_processing_pipeline()

    # # using tensorboard for observability
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Defining callback objects
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
        tf.keras.callbacks.ModelCheckpoint(filepath='./trained_model/retrieval/weights/trained_n_personalized_model',
                                           save_weights_only=True,
                                           save_best_only=True,
                                           save_freq="epoch",),
        # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    # instantiate the Final model class
    model = FinalModel(
        [128, 64, 32], # dense layer sizes - representing number of neurons per layer
        use_timestamp=True,
        movietitle_genres=features['movietitle_genres'],
        unique_user_ids=features['user_ids'],
        timestamp=features['timestamp'],
        timestamp_bucket=features['timestamp_bucket'],
        unique_movie_titles=features['movie_titles'],
        unique_genres=features['genres']

        )
    optimize = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    model.compile(optimizer=optimize)
    logger.info('model compiled successfully...')

    with tf.device('/GPU:0'): # setting tensorflow to run the fit operation on GPU
        model.fit(cached_train, epochs= 3, validation_data=cached_val, callbacks= callbacks)

        logger.info('model weights saved.')


if __name__ == "__main__":
    run_model_training()
