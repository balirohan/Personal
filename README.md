# RAG Project: Retrieval-Augmented Generation with Streamlit, FAISS, and GPT-3.5-Turbo

## UI

### Dark Mode
![image](https://github.com/user-attachments/assets/566885df-a539-4cbe-bb05-4ed421e168ea)

### Light Mode
![image](https://github.com/user-attachments/assets/eb84ad03-0de9-404f-8098-1b83a4f02f61)

## Overview
This project demonstrates a Retrieval-Augmented Generation (RAG) system, integrating Streamlit for a user-friendly interface, FAISS for efficient document retrieval, and OpenAI's GPT-3.5-turbo model for generating responses. The RAG setup enhances the model's capability by retrieving relevant documents, enabling it to provide more accurate and contextually relevant answers.

## Features
- User-friendly Interface: Built using Streamlit, offering an interactive environment for querying.
- Efficient Document Retrieval: Utilizes FAISS (Facebook AI Similarity Search) to index and retrieve relevant documents efficiently.
- GPT-3.5-turbo Integration: Leverages the OpenAI API to generate responses based on retrieved context.
- Scalable and Adaptable Design: The modular design allows easy updates to the retrieval or generation components.
## Demo

### Installation
**Prerequisites**

Ensure you have the following installed:

- Python 3.7+
- Streamlit
- FAISS
- OpenAI Python package

You also need an OpenAI API Key for accessing the GPT-3.5-turbo model.

## Steps
1. Clone the repository:

        git clone https://github.com/yourusername/your-repo-name.git
        cd your-repo-name

2. Install dependencies:

        pip install -r requirements.txt

3. Set up your OpenAI API Key:

- Create a .env file in the root directory and add your OpenAI API key as follows:

        OPENAI_API_KEY=your_openai_api_key

4. Run the Streamlit application:

        streamlit run app.py

## Usage
1. Open the app at http://localhost:8501 in your browser.
2. Input your query in the provided text box.
3. The app will retrieve relevant documents using FAISS and generate a response based on the retrieved context.

## Key Components
- Streamlit UI: A clean and intuitive interface for interacting with the model.
- FAISS Retrieval System: Handles fast similarity search to bring up the most relevant documents.
- GPT-3.5-turbo: Used for generating the final response based on the retrieved data.

## Future Enhancements
- Adding multi-language support for a broader audience.
- Improving response quality by experimenting with model parameters.
- Expanding the FAISS index with more documents for greater accuracy.

## License
This project is licensed under the MIT License. See LICENSE for more details.

## Acknowledgments
- [Streamlit](https://streamlit.io/) for providing a quick way to build web applications.
- [FAISS](https://faiss.ai/) for efficient document retrieval.
- [OpenAI](https://openai.com/) for their powerful GPT-3.5-turbo model.
