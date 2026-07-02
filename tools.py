from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

@tool
def get_weather(location:str)->str:
    """Get the weather at a location"""
    return f"Its sunny in {location}"


llm = HuggingFaceEndpoint(
    model='openai/gpt-oss-20b',
)
chat = ChatHuggingFace(llm=llm)

# model_with_tools = chat.bind_tools([get_weather])

# message = [{'role':'user','content':'What is the weather in Bangalore'}]

# response = model_with_tools.invoke(message)
# message.append(response)
# print(message)
# print('')

# for tool_call in response.tool_calls:
#     tool_result = get_weather.invoke(tool_call)
#     message.append(tool_result)

# print(message)
# print('')

# final_response = model_with_tools.invoke(message)
# print(final_response.text)

# Now doing it with bulit-in method create_agent 

agent = create_agent(
    model=chat,
    tools=[get_weather],
    system_prompt='You are a helpful assistant'
)

response = agent.invoke({'messages':[{'role':'user','content':'What the weather in Bangalore'}]})
print(response['messages'][-1].content)

