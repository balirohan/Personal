import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage

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
You're a smart chatbot that answers user-generated prompts only based on the context given to you.
You don't make any assumptions.
context:{context}
question:{question}
"""
prompt = PromptTemplate.from_template(template=question_template)

# Custom CSS for styling and loading FontAwesome
st.markdown(
    """
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">

    <!-- Preconnect to Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

    <!-- Load the Roboto font -->
    <link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900&display=swap" rel="stylesheet">

    <style>
    /* Background color for the app */
    .main {
        background-color: #333 !important;
    }

    /* Hide the Streamlit header */
    .stApp > header {
        display: none !important;
    }

    /* Title styling */
    h1 {
        color: #ff6f61 !important;
        text-align: center !important;
        font-family: 'Roboto', sans-serif !important;
        margin-bottom: 2rem !important;
    }

    /* Customize the upload button */
    .stFileUploader label {
        background-color: #ff6f61 !important;
        color: white !important;
        padding: 10px !important;
        border-radius: 10px !important;
        text-align: center !important;
        font-family: 'Roboto', sans-serif !important;
    }

    /* Text input box styling */
    .stTextInput input {
        padding: 10px !important;
        font-size: 1.1rem !important;
        font-family: 'Roboto', sans-serif !important;
        color: white !important;
    }

    /* Chat history styling */
    .stMarkdown p {
        font-family: 'Roboto', sans-serif !important;
        font-size: 1rem !important;
        color: white !important;
    }

    /* Separator line between chat messages */
    .stMarkdown hr {
        border: 1px solid #ff6f61 !important;
    }

    /* Button styling */
    button {
        background-color: #333 !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        font-family: 'Roboto', sans-serif !important;
    }

    button:hover {
        background-color: #ff6f61 !important;
        color: #fff !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px !important;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1 !important;
    }

    ::-webkit-scrollbar-thumb {
        background: #ff6f61 !important;
        border-radius: 10px !important;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #ff856f !important;
    }

    /* Reversed chat history display */
    div[data-testid="stChatMessageContainer"] {
        display: flex !important;
        flex-direction: column-reverse !important;
    }
    </style>

    <script>
    document.querySelectorAll('.stTextInput input').forEach(input => {
        input.setAttribute('autocomplete', 'off');
    });
    </script>

    """,
    unsafe_allow_html=True
)

# Streamlit UI with icon and animations
st.markdown(
    """
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">

    <style>
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1);
        }
    }

    .fas.fa-robot {
        animation: pulse 2s infinite;
        color: #ff6f61;
    }

    </style>

    <h1 style="text-align: center; color: #ff6f61;">
        <i class="fas fa-robot"></i> RAGbot
    </h1>
    """,
    unsafe_allow_html=True
)

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
user_input = st.text_input(
    label=":blue[Let us solve your queries!]",
    type="default",
    placeholder="Enter your prompt here",
    autocomplete=None,
    key="user_input"
)

if user_input:
    if st.session_state.retriever:
        # Show spinner while generating response
        with st.spinner("Generating response..."):
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

            # Clear the text input after processing
            st.session_state.user_input = ""

    else:
        st.error("Please upload a PDF file first to enable query processing.")

# Display chat history near input area
if st.session_state.history:
    st.markdown("## Chat History")

    # Iterate over history to show latest first
    for user_query, bot_response in st.session_state.history:
        st.write(f"**User:** {user_query}")
        st.write(f"**Bot:** {bot_response}")
        st.markdown("---")  # Insert a separator between chat entries
