import os
from dotenv import load_dotenv
import streamlit as st # type: ignore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage

# ------ MODEL CODE ------ 

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
        display: flex;
        align-items: center;
        justify-content: center;
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
    .stTextInput > input {
        padding: 10px !important;
        font-size: 1.1rem !important;
        font-family: 'Roboto', sans-serif !important;
        color: white !important;
        outline: none !important;
    }

    /* Placeholder text styling */
    ::placeholder {
        color: #ff6f61 !important;
        opacity: 1 !important; /* Ensure full opacity */
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
        background-color: #ff6f61 !important;
        color: white !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        font-family: 'Roboto', sans-serif !important;
    }

    button:hover {
        background-color: #cc594e !important;
        color: #fff !important;
    }

    /* Style the file remove button using data-testid */
    [data-testid="baseButton-minimal"] {
        background: none !important;
        border: none !important;
        color: #ff6f61 !important;
        font-size: 1.2rem !important;
        cursor: pointer !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    [data-testid="baseButton-minimal"]:hover {
        color: #cc594e !important;
        background: none !important;
    }

    [data-testid="stTextInput-RootElement"] {
        border: 2px solid #333 !important;
        border-radius: 8px !important;
        transition: border-color 0.3s ease !important;
    }

    [data-testid="stTextInput-RootElement"] input {
        padding: 10px !important;
        font-size: 1.1rem !important;
        font-family: 'Roboto', sans-serif !important;
        color: white !important;
        outline: none !important;
        border: none !important; /* Removing the border from the input itself */
        border-radius: inherit !important;
    }

    [data-testid="stTextInput-RootElement"]:focus-within {
        border-color: #ff6f61 !important;
        box-shadow: 0 0 2px rgba(255, 105, 97, 0.1) !important;
    }

    [data-testid="stTextInput-RootElement"]:hover {
        border-color: #ff6f61 !important;
    }

    [data-testid="stHeaderActionElements"] {
        display: none;
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

    .st-emotion-cache-15hul6a.ef3psqc13:nth-child(1) {
        /* Add your custom styles here */
        padding: 14px 12px 6px 12px !important;
        position: absolute;
        top: -255px !important;
        right: -100px !important;
    }
    </style>
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
            transform: scale(1.105);
        }
        100% {
            transform: scale(1);
        }
    }

    .fas.fa-robot {
        animation: pulse 2s infinite;
        color: #ff6f61;
        margin-right: 1rem;
        margin-bottom: 0.5rem;
    }

    h6 {
        margin-bottom: -40px !important;
    }

    </style>

    <h1 style="text-align: center; color: #ff6f61;">
        <i class="fas fa-robot"></i> RAGbot
    </h1>

    <h6>Hi there, I am RAGbot. I will give answers to any queries you might have once you give me a document!</h6>
    """,
    unsafe_allow_html=True
)

# Initialize the theme session state variable
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Function to toggle the theme
def toggle_theme():
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

# Create a variable to store the button icon
theme_icon = "☀️" if st.session_state.theme == "light" else "🌙"

# Create the theme toggle button
if st.button(theme_icon, key='top-right', on_click=toggle_theme):
    pass


# Update the CSS to reflect the new theme
if st.session_state.theme == "light":
    st.markdown(
        """
        <style>
        .main {
            background-color: #333 !important;
        }
        .stTextInput > input {
            color: white !important;
        }
        .stMarkdown p {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .main {
            background-color: #e9ecef !important;
            color: #333;
        }
        .stTextInput > input {
            color: #333 !important;
        }
        .stMarkdown p {
            color: #333 !important;
        }
        [data-testid="stNotification"] {
            background-color: #e66457;
            color: #fff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def delete_file_callback():
    if 'temp_file_path' in st.session_state and st.session_state.temp_file_path is not None:
        st.session_state.disable_input = True

        temp_file_path = st.session_state.temp_file_path
        file_name = os.path.basename(temp_file_path)  # Capture the file name

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        st.session_state.temp_file_path = None
        st.session_state.retriever = None

        # Set a flag in session state to show the delete notification
        st.session_state.file_deleted = True
        st.session_state.deleted_file_name = file_name

        st.session_state.disable_input = False

# File upload
uploaded_file = st.file_uploader(label=":blue[Upload a PDF file]", type=["pdf"], on_change=delete_file_callback)

# Display the appropriate notification
if 'file_deleted' in st.session_state and st.session_state.file_deleted:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; color: #28a745; margin-bottom: 0.35rem; margin-left: 1.5rem;">
            <i class='fas fa-check-circle' style="font-size: 1rem; margin-right: 0.5rem;"></i>
            <span>File '{st.session_state.deleted_file_name}' deleted successfully!</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Reset the flag after displaying the notification
    st.session_state.file_deleted = False
    st.session_state.deleted_file_name = None

elif uploaded_file:
    with st.spinner("Processing the uploaded PDF..."):
        st.session_state.disable_input = True

        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Store the temporary file path in session state
        st.session_state.temp_file_path = temp_file_path

        # Clear previous vector store
        st.session_state.retriever = None

        # Load new vector store
        st.session_state.retriever = load_vector_store(temp_file_path)

        # Show success notification for file upload
        st.markdown(
            f"""
            <div id="success-message" style="display: flex; align-items: center; color: #28a745; margin-bottom: 0.35rem; margin-left: 1.5rem;">
                <i class='fas fa-check-circle' style="font-size: 1rem; margin-right: 0.5rem;"></i>
                <span>File '{uploaded_file.name}' uploaded successfully!</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.session_state.disable_input = False

# Session state to store chat history and retriever
if 'history' not in st.session_state:
    st.session_state.history = []

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Initialize the disable_input session state variable
if 'disable_input' not in st.session_state:
    st.session_state.disable_input = False

# User input with customized placeholder
st.markdown(
    """
    <style>
    .stTextInput input::placeholder {
        color: #ff7d71 !important;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.session_state.disable_input:
    st.text_input(
        label=":blue[Let me solve your queries!]",
        type="default",
        placeholder="Enter your prompt here...",
        autocomplete=None,
        key="user_input_disabled",
        disabled=True
    )
else:
    user_input = st.text_input(
        label=":blue[Let me solve your queries!]",
        type="default",
        placeholder="Enter your prompt here...",
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

    else:
        st.error("Please upload a PDF file first to enable query processing.")

# Display chat history near input area
if st.session_state.history:
    st.markdown("## Chat History")

    # Iterate over history to show latest first
    for user_query, bot_response in st.session_state.history:
        st.write(f"**You:** {user_query}")
        st.write(f"**RAGbot:** {bot_response}")
        st.markdown("---")  # Insert a separator between chat entries
