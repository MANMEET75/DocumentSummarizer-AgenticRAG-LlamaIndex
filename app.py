from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import nest_asyncio
nest_asyncio.apply()
from llama_index.core import SimpleDirectoryReader
import nest_asyncio
nest_asyncio.apply()
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAI
import os
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub
os.environ['GOOGLE_API_KEY'] =  'AIzaSyCjyyhW36eS4Tkk6N2gITsBDOR6Q9kJeRI'
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector




app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIRECTORY = "data"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Uploads a PDF file to the server and replaces the existing file in the specified directory.
    """
    # Check if the file is a PDF
    if file.filename.endswith(".pdf"):
        # Remove existing file if it exists
        for existing_file in os.listdir(UPLOAD_DIRECTORY):
            file_path = os.path.join(UPLOAD_DIRECTORY, existing_file)
            os.remove(file_path)
        
        # Save the new file
        file_content = await file.read()
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        print(file_path)
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)

        # Using OpenAI (Useful in the case of big pdfs)
        # Settings.llm = OpenAI(model="gpt-3.5-turbo")
        # Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")


        # Using Hugging Face (Useful in the case of big pdfs)
        # Settings.llm = HuggingFaceHub(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",huggingfacehub_api_token="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")
        # Settings.embed_model = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")


        # Google Palm (Useful in the case of big pdfs)
        Settings.llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key="", temperature=0.1)
        Settings.embed_model = GooglePalmEmbeddings()

        summary_index = SummaryIndex(nodes)
        vector_index = VectorStoreIndex(nodes)

        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        vector_query_engine = vector_index.as_query_engine()



        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description=(
                "Useful for summarization questions related to MetaGPT"
            ),
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=(
                "Useful for retrieving specific context from the MetaGPT paper."
            ),
        )

        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                summary_tool,
                vector_tool,
            ],
            verbose=True
        )

        response = query_engine.query("What is the summary of the document?")
        final_response=str(response)
        print(final_response)




        return {"filename": file.filename, "Summary": final_response}
    else:
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF")

@app.get("/")
async def main(request: Request):
    """
    Displays the file upload form.
    """
    return templates.TemplateResponse("upload_form.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
