# Import the required libraries
import databutton as db
import streamlit as st
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
import os
import random
import textwrap as tr
from pocketsphinx import LiveSpeech
import pyttsx3

#engine = pyttsx3.init(driverName='espeak', debug=True)

# Helper functions (You can access them via View Code page of the app)
from text_load_utils import parse_txt, text_to_docs, parse_pdf, load_default_pdf
from df_chat import user_message, bot_message

# Initialize the Cohere API
cohere_api_key = db.secrets.get(name="vrda")

st.title("Chat with research paper / clinical study data 🤖")
st.info(
    "For your personal data! Powered by [cohere](https://cohere.com) + [LangChain](https://python.langchain.com/en/latest/index.html) + [Databutton](https://www.databutton.io) "
)

opt = st.radio("--", options=["Try the demo!", "Upload-own-file"])

# Initialize the speech recognition and text-to-speech engines
#recognizer = sr.Recognizer()
#engine = pyttsx3.init()

# Loading files
pages = None
if opt == "Upload-own-file":
    # Upload the files
    uploaded_file = st.file_uploader(
        "**Upload a pdf or txt file :**",
        type=["pdf", "txt"],
    )
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            doc = parse_txt(uploaded_file)
        else:
            doc = parse_pdf(uploaded_file)
        pages = text_to_docs(doc)
else:
    st.markdown(
        "Demo PDF : Paper on Machine Learning Algorithm Validation. "
        "[Link to download](https://www.researchgate.net/publication/346126028_Machine_Learning_Algorithm_Validation#fullTextFileContent)"
    )
    st.text("Quick Prompts to try (English | Croatian):")

    st.code("What is the meaning of this clinical study?")
    st.code("Koja je bit ove kliničke studije?")
    #pages = db.storage.text.get(key="steve-jobs-commencement-txt")
    pages = load_default_pdf()

voice_input = st.button("Voice Input")

page_holder = st.empty()
# Create our own prompt template
prompt_template = """Role: Medical Assistant that is an expert in reading Clinical study papers and documentation

Text: {context}

Question: {question}

Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available."""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# Bot UI dump
# Session State Initiation
prompt = st.session_state.get("prompt", None)

if prompt is None:
    prompt = [{"role": "system", "content": prompt_template}]

# If we have a message history, let's display it
for message in prompt:
    if message["role"] == "user":
        user_message(message["content"])
    elif message["role"] == "assistant":
        bot_message(message["content"], bot_name="Multilingual Personal Chat Bot")

if pages:
    # if uploaded_file.name.endswith(".txt"):

    # else:
    #     doc = parse_pdf(uploaded_file)
    # pages = text_to_docs(doc)

    with page_holder.expander("File Content", expanded=False):
        pages
    embeddings = CohereEmbeddings(
        model="multilingual-22-12", cohere_api_key=cohere_api_key
    )
    store = Qdrant.from_documents(
        pages,
        embeddings,
        location=":memory:",
        collection_name="my_documents",
        distance_func="Dot",
    )
    messages_container = st.container()
    question = st.text_input(
        "", placeholder="Type your message here", label_visibility="collapsed"
    )

    if st.button("Run", type="secondary"):
        prompt.append({"role": "user", "content": question})
        chain_type_kwargs = {"prompt": PROMPT}
        with messages_container:
            user_message(question)
            botmsg = bot_message("...", bot_name="Multilingual Personal Chat Bot")

        qa = RetrievalQA.from_chain_type(
            llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
            chain_type="stuff",
            retriever=store.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )

        answer = qa({"query": question})
        result = answer["result"].replace("\n", "").replace("Answer:", "")
        # with st.expander("Latest Content Source", expanded=False):
        #     sources = answer["source_documents"]
        # Update
        with st.spinner("Loading response .."):
            botmsg.update(result)
        # Add
        prompt.append({"role": "assistant", "content": result})

    st.session_state["prompt"] = prompt
else:
    st.session_state["prompt"] = None
    st.warning("No file found. Upload a file to chat!")