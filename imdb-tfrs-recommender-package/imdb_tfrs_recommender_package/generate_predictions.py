from typing import Dict,Any,Union
import faiss
from imdb_tfrs_recommender_package.processing import data_manager as dm

# FAISS Retrival class
class FaissRetrievalIndex():
    """ Using FAISS for approximate retrieval operation 
    
    * args:
    embedding_dimension: The dimension size of the vectors

    * model: This is the already compiled and trained_model
    """

    # list to store movie ids. This will be used to retrieve the movie names later
    movie_ids_list = []

    def __init__(self, embedding_dimension, model):
        self.embedding_dimension = embedding_dimension
        self.movie_model = model.candidate_model
        self.query_model = model.query_model
        # creating a distance based indices
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)


    # a function to index movie embeddings in FAISS
    def index_movie_in_faiss(self, movies, movie_ids):
        """ function that index movie embeddings

        args:
        * movies -  a batch of movie title list derived from a tensor dataset

        * movie_ids - the ids of the batch of movie title list
        
        """
        # retrieve movie embeddings
        movie_embeddings = self.movie_model(movies)

        # convert to a numpy array
        movie_embeddings_np = movie_embeddings.numpy()

        # add the movie embeddings to the faiss index
        self.faiss_index.add(movie_embeddings_np)

        # Update the movie ID list (ensure the order is consistent)
        movie_ids_list.extend(movie_ids)  # Add the movie IDs of the current batch

    
    # function to perform similarity search
    def search_top_k(self, user_id, k):
        """ perform similarity search of the query with the existing embeddings
        
        args:
        * user_id: the id which represents the query we hope to find a result for

        * k: the number of possible result to be returned
        """

        # get the user embeddings and convert it to numpy array
        user_embeddings = self.query_model(user_id)
        user_embeddings_np = user_embeddings.numpy()

        # performing search in faiss index
        distances, indices = self.faiss_index.search(user_embeddings_np, k)

        # Convert indices to movie IDs using the mapping
        recommended_movie_ids = []
        for index in indices:
            # print(index)
            movie_id_for_each_index = [movie_ids_list[i] for i in index] # Retrieve movie IDs for each index
            recommended_movie_ids.append(movie_id_for_each_index)

        return distances, recommended_movie_ids
    

def make_prediction(*, data: Union[Dict[str, Any], list[Dict[str, Any]]], embed_dim, model):
    """call this method to load recommendations for a user or group of users"""

    # create an instance of the Faiss retrieval index
    faiss_retrieval_index = FaissRetrievalIndex(embedding_dimension=embed_dim, model=model)


    # loading and handling our user data
    user_df_tensor = dm.load_and_preprocess_dataset_prod(data=data)


    user_df_tensor_map =  user_df_tensor.map(lambda x:
        {
        'movieID': x['movieID'],
        'originalTitle': x['originalTitle'],
        'genres': x['genres']
        }
    )

    for movie in user_df_tensor_map.batch().as_numpy_iterator():
        movie_ids = [mov for mov in movie['movieID']]  # return the movie ids for the batch
        faiss_retrieval_index.index_movie_in_faiss(movie, movie_ids)

    # searching for the top k most similar movies for a user
    user = 
    distances, recommended_movie_ids = faiss_retrieval_index.search_top_k(user, k=50)

    # removing duplicates from result
    final_movie_recommended_ids = list(dict.fromkeys(recommended_movie_ids[0]))

    # converting byte strings to string
    recommended_movie_str =[val.decode(encoding='utf-8') for val in final_movie_recommended_ids]

    recommended_movies = [df[df['movieID']== id]['originalTitle'].values[0] for id in recommended_movie_str]

    # Print the recommended movies for the user
    print(f"Recommended Movies: {recommended_movies}")