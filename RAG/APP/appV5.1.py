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
import tempfile
import time

# Load API key from environment variable
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize model and other components
Model = 'gpt-3.5-turbo'
model = ChatOpenAI(api_key=API_KEY, model=Model, temperature=0.2)

# Initialize session state to manage vector store
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

def load_vector_store(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    file_loader = PyPDFLoader(temp_file_path)
    page = file_loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pages = splitter.split_documents(page)
    vector_storage = FAISS.from_documents(pages, OpenAIEmbeddings())

    os.remove(temp_file_path)  # Clean up the temporary file

    return vector_storage.as_retriever()

# Streamlit UI
st.title("RAGbot")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf", help="Click here to upload a PDF document")

if uploaded_file:
    # Update vector store with the new PDF
    with st.spinner("Processing PDF..."):
        start_time = time.time()
        st.session_state.retriever = load_vector_store(uploaded_file)
        end_time = time.time()
        st.success(f"PDF uploaded and vector store updated successfully in {end_time - start_time:.2f} seconds!")

# Load vector store if not already done
if st.session_state.retriever is None:
    st.warning("Please upload a PDF document to begin.")
else:
    # Define the prompt template
    question_template = """
    You're a smart chatbot that answers user generated prompts only based on the context given to you.
    You don't make any assumptions.
    context:{context}
    question:{question}
    """
    prompt = PromptTemplate.from_template(template=question_template)

    # Session state to store chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        # Retrieve context and generate response
        with st.spinner("Generating response..."):
            relevant_docs = st.session_state.retriever.get_relevant_documents(user_input)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Create the prompt using the template
            formatted_prompt = prompt.format(context=context, question=user_input)

            # Convert prompt to message format for the model
            messages = [HumanMessage(content=formatted_prompt)]

            # Get the response from the model
            start_time = time.time()
            response = model(messages=messages)
            end_time = time.time()

            # The response is a single AIMessage object, so we directly access its content
            ai_message = response.content.strip()

            # Update chat history (insert at the beginning to display latest first)
            st.session_state.history.insert(0, (user_input, ai_message))
            st.success(f"Response generated in {end_time - start_time:.2f} seconds!")

    # Display chat history near input area
    if st.session_state.history:
        st.markdown("## Chat History")

        # Iterate over history in reverse order to show latest first
        for idx in range(len(st.session_state.history) - 1, -1, -1):
            user_query, bot_response = st.session_state.history[idx]
            st.write(f"**User:** {user_query}")
            st.write(f"**Bot:** {bot_response}")

            # Insert a separator between chat entries, except for the last one
            if idx > 0:
                st.markdown("---")

        # Automatically scroll to the latest chat entry
        st.markdown('<style>div[data-testid="stHorizontalBlock"][role="scrollbar"] {height: auto !important;} </style>', unsafe_allow_html=True)
        st.markdown('<style>div[data-testid="stHorizontalBlock"][role="scrollbar"] {height: 400px !important;} </style>', unsafe_allow_html=True)
