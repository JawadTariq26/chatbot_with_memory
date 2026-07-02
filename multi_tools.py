from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.tools import tool
from langchain_tavily import TavilySearch
import os
from langchain.agents import create_agent, structured_output
from pydantic import Field,BaseModel
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents.middleware import SummarizationMiddleware


load_dotenv()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

@tool
def get_weather(location:str)->str:
    """Get the weather at a location"""
    return f"Its sunny in {location}"


@tool('calculator', description='Performs arithematic calculation. Use this for any math problems')
def calcu(expression: str) -> str:
    """Evaluate mathematical expressions """
    return str(eval(expression))

llm = HuggingFaceEndpoint(
    model='openai/gpt-oss-20b',
)
chat = ChatHuggingFace(llm=llm)

class Structured(BaseModel):
    title : str
    description : str

tool_result = TavilySearch(
    max_results=5,
    topic='general'
)

agent = create_agent(
    model=chat,
    checkpointer= InMemorySaver(),
    middleware=[SummarizationMiddleware(chat,trigger=('messages',10),keep=('messages',4))],
    # system_prompt="""
    # You are a helpful assistant.

    # Use calculator for ALL arithmetic calculations.
    # Never calculate yourself.
    # """
)
configure = {'configurable':{'thread_id':'test1'}}


questions = [
    'What is 2+2?',
    'What is 2+8?',
    'What is 3*7?',
    'What is 6-8?',
    'What is 3*3?',
    'What is 5/5?'
]
for q in questions:
    response = agent.invoke({'messages':[HumanMessage(q)]},config=configure)
    print(f'Messages:{response}')
    print(f'Messages :{len(response['messages'])}')

# user_input = 'What is the current AI news and then calculate 5 + 5 '
# for step in agent.stream({'messages': [{'role':'user','content':user_input}]},stream_mode='values'):
#     print(step['messages'][-1])

# print(agent.invoke({'messages':[HumanMessage(content="hey my name is jawad")]},config=config))
# r = agent.invoke({'messages':[HumanMessage(content="hey whats my name?")]},config=config)
# print(r['messages'][-1].content)