import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load API key from environment variable
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize model and other components
Model = 'gpt-3.5-turbo'
model = ChatOpenAI(api_key=API_KEY, model=Model, temperature=0.2)

@st.cache_resource(show_spinner=False, ttl=3600)
def load_vector_store(file_path):
    file_loader = PyPDFLoader(file_path)
    page = file_loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pages = splitter.split_documents(page)
    vector_storage = FAISS.from_documents(pages, OpenAIEmbeddings())
    return vector_storage.as_retriever()

# Define the prompt template
question_template = """
You're a smart chatbot that answers user generated prompts only based on the context given to you.
You don't make any assumptions.
context:{context}
question:{question}
"""
prompt = PromptTemplate.from_template(template=question_template)

# Streamlit UI
st.title("RAGbot")

# File upload
uploaded_file = st.file_uploader(label=":blue[Upload a PDF file]", type=["pdf"])

# Session state to store chat history and retriever
if 'history' not in st.session_state:
    st.session_state.history = []

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Process the uploaded file
if uploaded_file:
    with st.spinner("Processing the uploaded PDF..."):
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Clear previous vector store
        st.session_state.retriever = None

        # Load new vector store
        st.session_state.retriever = load_vector_store(temp_file_path)

# User input
user_input = st.text_input(label=":blue[Let us solve your queries!]", placeholder="Enter your prompt here")

if user_input and st.session_state.retriever:
    # Retrieve context and generate response
    relevant_docs = st.session_state.retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Create the prompt using the template
    formatted_prompt = prompt.format(context=context, question=user_input)

    # Convert prompt to message format for the model
    messages = [HumanMessage(content=formatted_prompt)]

    # Get the response from the model
    response = model(messages=messages)

    # The response is a single AIMessage object, so we directly access its content
    ai_message = response.content.strip()

    # Update chat history (insert at the beginning to display latest first)
    st.session_state.history.insert(0, (user_input, ai_message))

# Display chat history near input area
if st.session_state.history:
    st.markdown("## Chat History")

    # Iterate over history to show latest first
    for user_query, bot_response in st.session_state.history:
        st.write(f"**User:** {user_query}")
        st.write(f"**Bot:** {bot_response}")
        st.markdown("---")  # Insert a separator between chat entries

# Automatically scroll to the latest chat entry
st.markdown(
    '''
    <style>
        div[data-testid="stChatMessageContainer"] {
            display: flex;
            flex-direction: column-reverse;
        }
    </style>
    ''',
    unsafe_allow_html=True
)
