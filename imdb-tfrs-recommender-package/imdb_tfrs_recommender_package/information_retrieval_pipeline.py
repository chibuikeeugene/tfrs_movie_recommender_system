import tensorflow as tf
import tensorflow_recommenders as tfrs
from  imdb_tfrs_recommender_package.processing import model_manager

# using the tfrs.Model class to wrap our two-tower model architecture and define metrics and loss functions

class FinalModel(tfrs.models.Model):
    def __init__(self, layer_sizes, use_timestamp, movietitle_genres):
        super().__init__()
    
        self.query_model: tf.keras.Model = model_manager.QueryModel(layer_sizes, use_timestamp)
        self.candidate_model: tf.keras.Model = model_manager.CandidateModel(layer_sizes)
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








def save_retrieval_model():
    """save the """
    pass


def remove_old_model(model:):
    pass