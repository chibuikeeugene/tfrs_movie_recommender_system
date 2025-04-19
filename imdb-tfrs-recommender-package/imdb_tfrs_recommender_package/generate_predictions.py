import faiss

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
    

def predict():
    pass