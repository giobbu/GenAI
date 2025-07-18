from llama_index.core import PromptTemplate 
from loguru import logger
import streamlit as st

def run_retriever(index, user_query, params):
    """Retrieve relevant nodes from the index based on the user query."""     
    retriever = index.as_retriever(similarity_top_k=params.similarity_top_k)
    nodes = retriever.retrieve(user_query)

    st.header("1 Task di RAG - Estrazione di Testo")
    for i, node in enumerate(nodes):
        st.subheader(f"Testo Estratto {i+1}")
        logger.info(f"Node: {node.text}")
        st.write(f"Node: {node.text}")

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
    return fmt_qa_prompt