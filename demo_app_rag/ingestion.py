
from loguru import logger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def ingestion_documents(params):
    """Ingest documents from a directory and create a vector store index."""
    documents = SimpleDirectoryReader(input_dir=params.directory, required_exts=params.required_exts).load_data()
    logger.info(f"Number of documents loaded: {len(documents)}")
    parser = SentenceSplitter(chunk_size=params.chunk_size, chunk_overlap=params.chunk_overlap)
    nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    vector_index = VectorStoreIndex(
                        nodes=nodes, 
                        embed_model= HuggingFaceEmbedding(model_name=params.embed_model_name),
                        insert_batch_size=1000,
                        show_progress=True
                        )
    return vector_index