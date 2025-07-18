import streamlit as st
from llama_index.llms.ollama import Ollama
from setting import configRAG
from utils import StructuredOutput, create_document, generate_response
import json
import time

def main():
    st.title("Q&A RAG app dimostrativa")

    # Input query from user
    user_query = st.text_input("Domanda:", value=f"Che cos'è il DOCUMENTO DI LAVORO DEI SERVIZI DELLA COMMISSIONE?")

    if st.button("Risposta"):
        with st.spinner("Generating response..."):

            params = configRAG()

            from loguru import logger
            from llama_index.core import SimpleDirectoryReader
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.core import VectorStoreIndex
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.core.response_synthesizers import get_response_synthesizer

            directory = r"doc/"
            required_exts = [".txt"]
            documents = SimpleDirectoryReader(input_dir=directory, required_exts=required_exts).load_data()
            logger.info(f"Number of documents loaded: {len(documents)}")
            parser = SentenceSplitter(chunk_size=params.chunk_size, chunk_overlap=params.chunk_overlap)
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            index = VectorStoreIndex(
                                nodes=nodes, 
                                embed_model= HuggingFaceEmbedding(model_name=params.embed_model_name) ,
                                insert_batch_size=1000,
                                show_progress=True
                                )

            llm = Ollama(
                        model=params.llm_model_name,
                        request_timeout=params.request_timeout,
                        context_window=params.context_window,
                        json_mode=params.json_mode
                    )
            
            structured_llm = llm.as_structured_llm(StructuredOutput)
            
            retriever = index.as_retriever()
            nodes = retriever.retrieve(user_query)

            st.header("1 Task di RAG - Estrazione di Testo")
            for i, node in enumerate(nodes):
                st.subheader(f"Testo Estratto {i+1}")
                logger.info(f"Node: {node.text}")
                st.write(f"Node: {node.text}")

            from llama_index.core import PromptTemplate 

            qa_prompt = PromptTemplate(
                            """\
                        Context information is below.
                        ---------------------
                        {context_str}
                        ---------------------
                        Given the context information and not prior knowledge, answer the query.
                        Query: {query_str}
                        Answer: \
                        """
                        )
            context_str = "\n\n".join([r.get_content() for r in nodes])
            fmt_qa_prompt = qa_prompt.format(
                context_str=context_str, query_str=user_query
            )

            response = generate_response(fmt_qa_prompt, structured_llm)

            # formatted prompt
            st.subheader("Prompt Formattato")
            st.write(fmt_qa_prompt)

            logger.info(f"Response: {response}")
            st.header("2 Task di RAG - Generatore Sintetica della Risposta")
            st.write(response)

if __name__ == "__main__":
    main()
