import streamlit as st
from setting import configRAG
from ingestion import ingestion_documents
from retriever import run_retriever
from synthetizer import run_synthetizer
from loguru import logger
import json
import time

def main():
    st.title("Q&A RAG app dimostrativa")

    # Input query from user
    user_query = st.text_input("Domanda:", value=f"Che cos'è il DOCUMENTO DI LAVORO DEI SERVIZI DELLA COMMISSIONE?")

    if st.button("Risposta"):

        with st.spinner("Generating response..."):

            params = configRAG()

            index = ingestion_documents(params)

            fmt_qa_prompt = run_retriever(index, user_query)

            response = run_synthetizer(fmt_qa_prompt, params)

            # formatted prompt
            st.subheader("Prompt Formattato")
            st.write(fmt_qa_prompt)
            logger.info(f"Response: {response}")
            st.header("2 Task di RAG - Generatore Sintetica della Risposta")
            st.write(response)

if __name__ == "__main__":
    main()
