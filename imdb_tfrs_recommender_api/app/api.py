import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Any, Optional
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel
import pandas as pd

from imdb_tfrs_recommender_package.imdb_tfrs_recommender_package import generate_predictions
from imdb_tfrs_recommender_package.imdb_tfrs_recommender_package.config.core import DATASET, RETRIEVAL_MODEL
from imdb_tfrs_recommender_package.imdb_tfrs_recommender_package.processing import model_manager


# create an api router object
api_router = APIRouter()

# load the saved model
loaded_model = model_manager.load_model(
    model= RETRIEVAL_MODEL
)

# load the movies dataframe
movies_data_df =  pd.read_csv(f'{DATASET}/movies_data.csv')

# request class schema
class Request(BaseModel):
    """ request object schema"""
    userID: str = 'ur4592644',
    movieID: str = 'tt0120884',
    rating: int = 10,
    review_date: str = '16 January 2005',
    originalTitle: str =  'When the Light Comes',
    genres: str =  "Adventure,Drama,Romance",
    runtimeMinutes: int = 115


# response class schema
class PredictionResults(BaseModel):
    """response object schema"""
    version: str
    movies: Optional[list[str]]


@api_router.post('/predict', response_model = PredictionResults, status_code=200)
async def predict(*, data: Request):
    """make predictions with the saved model"""

    predictions = generate_predictions.make_prediction(
        data=jsonable_encoder(data),
        model=loaded_model,
        movies_table=movies_data_df
    )

    return predictions
