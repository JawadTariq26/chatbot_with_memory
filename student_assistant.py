from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os 
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from pydantic import BaseModel
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
import json

load_dotenv()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

#TOOLS
@tool('calculator', description='Performs arithematic calculation. Use this for any math problems')
def calculator(expression: str) ->str :
    """ Evaluate the given expression """
    return str(eval(expression))


@tool
def quiz_generator(topic: str, num_questions : int) -> str:
    """Generate a quiz on the given topic and follow the given schema"""
    return f"""
    Create {num_questions} MCQs on {topic}.
    
    Return only valid json:
    {{
    "topic":"{topic}",
    "questions":[
    {{
    "question" :'...',
    "option_a" :'...',
    "option_b" :'...',
    "option_c" :'...',
    "option_d" :'...',
    "correct_answer": 'A' 
    }}
    ]
    }}
    """

@tool
def study_notes_geneartor(topic: str) -> str:
    """Generate a short notes on the given topic for the students"""
    return f"""
    Create concise study notes on {topic}.

    Return only valid json:
    {{
    "topic":"{topic}",
    f"Notes on {topic}":[
    {{
    "Definition" :'...',
    "Key_concepts" :'...',
    "Important_points" :'...',
    "Short_summary" :'...',
    }}
    ]
    }}
    Keep the notes easy to understand for students.
    """


search_tool = TavilySearch(max_results=3, topic='general')


file_path = "memory.json"

if os.path.exists(file_path):
    try:
        with open(file_path,'r',encoding='utf-8') as f:
            memory = json.load(f)
            recent_history = memory['history'][-3:]
        
    except json.JSONDecodeError:
        memory = {'history': []}
        recent_history = []
else:
    memory = {'history':[]}
    recent_history = []

context = ''
for item in recent_history:
    context += f'User: {item['user_input']}\n'
    context += f'Assistant: {item['assistant']}\n\n'


llm = HuggingFaceEndpoint(
    model='openai/gpt-oss-20b',
)
chat = ChatHuggingFace(llm=llm)   # convert it into chat model

agent = create_agent(
    model= chat,
    tools= [calculator, quiz_generator, study_notes_geneartor, search_tool],
    checkpointer= InMemorySaver(),
    middleware=[SummarizationMiddleware(chat, trigger=('messages',10), keep=('messages',4))]
)
configure = {'configurable':{'thread_id':'student'}}


while True:
    user_input = input('Here am I for your help :')
    if user_input.strip().lower()=='exit':
        print('exiting the loop')
        break

    full_prompt = f"""
    Previous conversation history:
    {context}

    Current user question:
    {user_input}
    """

    response = agent.invoke({'messages': full_prompt},config=configure) #type:ignore

    ai_response = (response['messages'][-1].content)

    try:
        clean_response = ai_response.replace("```json", "").replace("```", "").strip() 
        stored_response = json.loads(clean_response)
    except json.JSONDecodeError:
        stored_response = ai_response
    
    memory["history"].append({'user_input' : user_input,
                              'assistant' : stored_response})

    with open(file_path,'w',encoding='utf-8') as f:
        json.dump(memory, f, indent=2)