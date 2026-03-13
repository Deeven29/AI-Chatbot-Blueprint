import streamlit as st
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.rag_utils import retrieve_document
from utils.web_search import search_web
from models.llm import get_chatgroq_model,get_openai_model,get_gemini_model


def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the chat model"""
    try:
        # Prepare messages for the model
        formatted_messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        prompt=messages[-1]["content"]
        context = retrieve_document(prompt)
        if context:
            source="📄Document Knowledge"
            augmented_prompt=f"Use this document information:\n{context}\n\nQuestion:{prompt}"
        else:
            web_info=search_web(prompt)
            source="🌐Web Search"
            augmented_prompt=f"Use this web information:\n{web_info}\n\nQuestion:{prompt}"
            
        formatted_messages.append(HumanMessage(content=augmented_prompt))
        
        
        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content,source
    
    except Exception as e:
        return f"Error getting response: {str(e)}","Error"

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## 🔧 Installation
                
    
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    
    You'll need API keys from your chosen provider. Get them from:
    
    ### OpenAI
    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    - Create a new API key
    - Set the variables in config
    
    ### Groq
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key
    - Set the variables in config
    
    ### Google Gemini
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Create a new API key
    - Set the variables in config
    
    ## 📝 Available Models
    
    ### OpenAI Models
    Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
    Popular models include:
    - `gpt-4o` - Latest GPT-4 Omni model
    - `gpt-4o-mini` - Faster, cost-effective version
    - `gpt-3.5-turbo` - Fast and affordable
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.1-70b-versatile` - Large, powerful model
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    - `gemini-1.5-flash` - Fast and efficient
    - `gemini-pro` - Standard model
    
    ## How to Use
    
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!
    
    ## Tips
    
    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities and costs
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh
    
    ## Troubleshooting
    
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status
    
    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)

def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")
    
    # Get configuration from environment variables or session state
    # Default system prompt
    system_prompt = ""
    
    
    # Determine which provider to use based on available API keys
    chat_model = get_chatgroq_model()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    response_mode=st.radio(
        "Response Mode",["Concise","Detailed"],
        horizontal=True
    )
    
    if response_mode=="Concise":
        system_prompt="Answer briefly and clearly."
    else:
        system_prompt="Provide a detailed and well explained answer."
    
    #upload document
    uploaded_file = st.file_uploader("Upload PDF Document",type=["pdf"])
    if uploaded_file:
        from utils.rag_utils import add_documents
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from PyPDF2 import PdfReader
        pdf_reader=PdfReader(uploaded_file)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks=splitter.split_text(text)
        add_documents(chunks)
        st.success("Document uploaded and indexed successfully!")
  
    # Chat input
    # if chat_model:
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                response,source = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)
                st.caption(f"Source:{source}")
        
        # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
 

def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
        
        # Add clear chat button in sidebar for chat page
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()