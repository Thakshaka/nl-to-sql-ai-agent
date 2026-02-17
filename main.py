import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import AzureChatOpenAI
from langchain_classic import hub
from langchain.agents import create_agent
# from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
AZURE_OPENAI_DEPLOYMENT_NAME =os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]
DB_PORT = os.environ["DB_PORT"]

db = SQLDatabase.from_uri(f'mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?ssl=true&ssl_mode=REQUIRED')

# initialize llm
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version=OPENAI_API_VERSION,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# initialize the toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# @tool
# def get_weather(location: str) -> str:
#     """Get weather information for a location."""
#     return f"Weather in {location}: Sunny, 72°F"

# tools.append(get_weather)

# prompt template for nl2sql
prompt_template = hub.pull('langchain-ai/sql-agent-system-prompt')

system_message = prompt_template.format(dialect=db.dialect, top_k=5)

summarization_middleware = SummarizationMiddleware(
    model=llm, 
    trigger=("tokens", 4000), 
    keep=("messages", 20)
)

checkpointer = InMemorySaver()

# create the ai agent
agent = create_agent(llm, tools, system_prompt=system_message, middleware=[summarization_middleware], checkpointer=checkpointer)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: ChatMessage
    thread_id: str

@app.post("/api/chat/completion")
async def chat_completion(chat_request: ChatRequest):
    config: RunnableConfig = {"configurable": {"thread_id": chat_request.thread_id}}
    response = agent.invoke({"messages": [{"role": chat_request.message.role, "content": chat_request.message.content}]}, config)
    final_answer = response['messages'][-1].content
    return {"answer": final_answer}

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
