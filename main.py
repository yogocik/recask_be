from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery 
from google.oauth2 import service_account
from pandas import DataFrame
from typing import Dict, List
from dotenv import load_dotenv 
from pydantic import BaseModel
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import re
import uvicorn

# Load environment variable from .env 
load_dotenv()

# Initiate FastAPI web-server
app = FastAPI()

origins = [
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initiate DB
credentials = service_account.Credentials.from_service_account_file('./secrets/svc_acc.json')
BQClient = bigquery.Client(
    credentials= credentials,
    project=credentials.project_id
)

# Initiate LLM resources
llm = VertexAI(model_name=os.getenv("GENAI_MODEL_NAME","gemini-pro"))
embedding_model = VertexAIEmbeddings(
            model_name= os.getenv("GENAI_EMBEDDING_MODEL_NAME","text-embedding-004"), 
            project= os.getenv("GCP_PROJECT_ID")
        )
BQVectorDB = BigQueryVectorStore(
            project_id=os.getenv("GCP_PROJECT_ID"),
            dataset_name=os.getenv("GCP_DATASET_ID"),
            table_name=os.getenv("GCP_TABLE_ID"),
            location=os.getenv("GCP_LOCATION"),
            embedding=embedding_model,
        )


def retrieve_from_vector_db(query:str, limit:int=10) -> List[Document]:
    docs = BQVectorDB.similarity_search(query=query,
                                        k=limit)
    return docs

def execute_query(query:str) -> DataFrame:
    # Execute the query and get the results
    query_job = BQClient.query(query)
    # Convert the result to a pandas DataFrame
    return query_job.to_dataframe()


def get_all_selected_movies() -> List[Dict[str,str]]:
    # Define the query
    query = """
        SELECT *
        FROM `data-engineering-360214.internal.clean_movies`
        WHERE id IN (SELECT id FROM `data-engineering-360214.internal.vector_movies`)
    """
    # Fetch data
    data = execute_query(query)
    #  Convert to dict and return
    res = [{**x,
            #'is_answer':False
            } for x in data.to_dict(orient='records')]
    return res


def get_paginated_movies(page:int = 1, limit:int = 10) -> List[Dict[str,str]]:
    # Define the query
    query = f"""
        SELECT * FROM (
        SELECT *
        FROM `data-engineering-360214.internal.clean_movies`
        WHERE id IN (SELECT id FROM `data-engineering-360214.internal.vector_movies`)
        ORDER BY id
        ) LIMIT {limit} OFFSET {(page-1)* limit}
    """
    # Fetch data
    data = execute_query(query)
    #  Convert to dict and return
    return data.to_dict(orient='records')

def get_movies_by_ids(id: str) -> List[Dict[str,str]]:
    print(f"Expected data with id : {id}")
    # Define the query
    query = f"""
        SELECT 
            * 
        FROM (
            SELECT *
            FROM `data-engineering-360214.internal.clean_movies`
            WHERE id IN (SELECT id FROM `data-engineering-360214.internal.vector_movies`)
        ) WHERE id IN ({id})
    """
    # Fetch data
    data = execute_query(query)
    #  Convert to dict and return
    return data.to_dict(orient='records')


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/all_movies")
def get_all_movies():
    return {'data': get_all_selected_movies()}

@app.get("/paginated_movies")
def get_all_paginated_movies(page:int = 1, 
                             limit: int=20):
    return {'data': get_paginated_movies(page=page, limit=limit)}

class MovieRequestById(BaseModel):
    id: List[str]

@app.get("/movie_by_id")
def get_movie(ids: str): # Comma-separated string
    return {'data': get_movies_by_ids(ids)}

class UserCustomQuery(BaseModel):
    query: str

@app.post("/ask_model") 
def get_query_answer(query: UserCustomQuery):
    # Search in vector db for enriched contents
    docs = retrieve_from_vector_db(query.query)
    doc_ids = ','.join(set([str(doc.metadata['id']) for doc in docs]))
    if len(docs) <= 0:
        return {"status": "UNIDENTIFIED",
                "data": []}
    # Take all items mentioned in enriched contents
    enriched_movies = get_movies_by_ids(doc_ids)
    # Combine with prompts (question + enrichments)
    prompt_template = """
    Context: \n```{context}```\n
    Request: Based on context above, {query}
    Provide the response strictly as python list of each ID only, with no additional text or explanations
    If there is no answer then return empty python list.
    """

    # Submit to LLM and return in JSON format
    prompt = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "query"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"context": enriched_movies, "query": query.query})
    match = re.search(r'\[(.*?)\]', response)
    split = [x.strip() for x in match.group(1).split(",")] if match else []
    print('LLM Response\n', response)
    selected = [{**data, 'is_answer': True} for data in enriched_movies if str(data['id']) in split]
    recommended = [{**data, 'is_answer': False} for data in enriched_movies if str(data['id']) not in split]
    data = [*selected, *recommended]
    data.sort(reverse=True,  key=lambda x: f"{(int(x['is_answer']))}{int(x['id'])}")
    return {"status": "OK", 
            # "response": response, 
            # "raw": enriched_movies,
            # "split": split,
            "data": data
        }

if __name__ == '__main__':
    uvicorn.run('main:app', 
                host=os.getenv("SERVER_HOST", '127.0.0.1'),
                port=os.getenv("SERVER_PORT", 4000),
                reload=True
                )
