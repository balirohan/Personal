import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Load environment variables
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
Model = 'gpt-3.5-turbo'

# Initialize the model
model = ChatOpenAI(api_key=API_KEY, model=Model, temperature=0.2)
parser = StrOutputParser()


# Define a function to format text for better readability
def format_text(text):
    formatted_text = text.replace('\n', '\n\n')
    return formatted_text


# Load and split PDF
file_loader = PyPDFLoader('Hitachi Manual.pdf')
page = file_loader.load_and_split()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
pages = splitter.split_documents(page)

# Initialize vector storage and retriever
vector_storage = FAISS.from_documents(pages, OpenAIEmbeddings())
retriever = vector_storage.as_retriever()

# Define the prompt template
question_template = """
You're a smart chatbot that answers user-generated prompts only based on the context given to you.
You don't make any assumptions.
context:{context}
question:{question}
"""
prompt = PromptTemplate.from_template(template=question_template)

# Initialize the chain
result = RunnableParallel(context=retriever, question=RunnablePassthrough())
chain = result | prompt | model | parser

# Custom CSS
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px; /* Added margin below title */
        }
        .user-message {
            border-radius: 20px;
            padding: 15px;
            margin-top: 10px;
        }
        .bot-message {
            border-radius: 20px;
            background-color: #333;
            padding: 13px;
            margin-top: 3px;
            margin-bottom: 10px; /* Added margin below bot message */
        }
        .text-input {
            margin-top: 20px; /* Added margin above text input */
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.markdown('<h1 class="title">RAGbot</h1>', unsafe_allow_html=True)

# Display chat history in reverse order
for message in reversed(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{message["text"]}</div>', unsafe_allow_html=True)

# User input
st.markdown('<div class="text-input">', unsafe_allow_html=True)
user_question = st.text_input("Ask a question:", key="input", placeholder="Type your question here...",
                              help="Type your question and click the button to get an answer.")
st.markdown('</div>', unsafe_allow_html=True)

# Retrieve context and generate response only if there's a user question
if user_question:
    response = chain.invoke(user_question)
    formatted_response = format_text(response)

    # Save messages at the beginning of the list
    st.session_state.messages.insert(0, {"role": "user", "text": user_question})
    st.session_state.messages.insert(0, {"role": "bot", "text": formatted_response})
