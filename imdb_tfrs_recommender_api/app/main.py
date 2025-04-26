from fastapi import FastAPI, APIRouter
import uvicorn
from fastapi.responses import HTMLResponse
from app.api import api_router


app  = FastAPI(
    title= 'Information Retrieval and Ranking Recommender System API',
    openapi_url= '/api/v1/openapi.json'
)

index_router =  APIRouter()

@index_router.get('/')
def index():
    """Basic html response"""

    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(body)

app.include_router(router=index_router)
app.include_router(router=api_router)

if __name__ == '__main__':
    uvicorn.run(app=app, host='localhost', port=8000)