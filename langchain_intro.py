from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()

llm = HuggingFaceEndpoint(
    model='mistralai/Mistral-7B-Instruct-v0.3',
)
chat = ChatHuggingFace(llm=llm)

response = chat.invoke('What is Artificial Intelligence in short?') #using invode

# response = chat.stream('What is artificial intelligence tell me in 300 words') #using stream
# for chunk in response:
#     print(chunk.text , end='', flush=True)

# response = chat.batch([
#     "What is Artificial Intelligence",
#     "What is Machine Learning",
#     "What is USA"
# ])
print(response.content)