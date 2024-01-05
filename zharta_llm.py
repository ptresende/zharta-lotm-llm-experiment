from llama_index import VectorStoreIndex, StorageContext, SimpleDirectoryReader, download_loader, load_index_from_storage
from llama_index.schema import Document
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent
from llama_index.prompts import PromptTemplate

import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader


st.set_page_config(page_title="Zharta & LOTM LLM", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

openai.api_key = st.secrets.openai_key

st.title("Zharta & The Otherside LLM 💬🦙")
st.info("This is an experimental feature! Also please don't use it a lot since I'm paying for the OpenAI requests 💸", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Zharta or the Otherside metaverse!"}
    ]


@st.cache_data(show_spinner=True)
def setup_base_data_query_engine():
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/zharta_lotm"
    )
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    return query_engine

query_engine_base_data = setup_base_data_query_engine()


@st.cache_data(show_spinner=True)
def setup_contacts_query_engine():
    documents_contacts = [
        Document(text="Zharta's Email Address is test@test.com"),
        Document(text="Zharta's X account is @zhartafinance"),
        Document(text="Zharta's Discord server is @randomcenas"),
    ]
    index_contacts = VectorStoreIndex.from_documents(documents_contacts)
    query_engine_contacts = index_contacts.as_query_engine()
    return query_engine_contacts

query_engine_contacts = setup_contacts_query_engine()


def setup_agent(query_engine_base_data, query_engine_contacts):
    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine_base_data,
            metadata=ToolMetadata(
                name="fundamental_data",
                description=(
                    "Provides information about Zharta and Otherside."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=query_engine_contacts,
            metadata=ToolMetadata(
                name="contacts",
                description=(
                    "Provides information about Zharta's contacts. "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
    ]

    context_agent = OpenAIAgent.from_tools(
        query_engine_tools,
        verbose=False,
    )

    return context_agent


agent = setup_agent(query_engine_base_data, query_engine_contacts)

template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = PromptTemplate(template)

context_str = (
    'Always answer as if you were a customer service agent working at Zharta. '
    'Prefer to use "we" instead of "them" when talking about Zharta, its team or its customer service. '
    'When asked about what is Zharta or what Zharta does, always keep in mind that Zharta provides two services: lending and borrowing using NFTs as collateral, and renting for the LOTM game. '
    'Never abbreviate Legens of the Mara as LOTM. '
    'Instead of saying [email protected] consult that information from your tools. '
    'Zharta does not currently have a token, but it is possible that we will have one in the future. '
    'Zharta has no current plans to do a token airdrop. '
    'Give simple and direct answers.'
)

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = agent

if user_query := st.chat_input("Your question"): # Prompt for user input and save to chat history
    prompt = qa_template.format(context_str=context_str, query_str=user_query)
    st.session_state.messages.append({"role": "user", "content": prompt, "user_query": user_query})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["user_query"])
        else:
            st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
