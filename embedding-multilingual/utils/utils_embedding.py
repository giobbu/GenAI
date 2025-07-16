from llama_index.finetuning import EmbeddingQAFinetuneDataset
from loguru import logger
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_qa_dataset(dataset_name):
    """
    Load the dataset and clean it by removing empty entries.
    """
    logger.info(f"Loading dataset from {dataset_name}")
    it_dataset = EmbeddingQAFinetuneDataset.from_json(dataset_name)
#    keys_to_delete = [key for key in it_dataset.corpus.keys() if it_dataset.corpus[key] == '']
#    for key in keys_to_delete:
#        del it_dataset.corpus[key]
    return it_dataset

def evaluate_embedding(dataset, embed_model, top_k=5, verbose=False):
    """ Evaluate the dataset using the provided embedding model and return evaluation results. 
    Args:
        dataset (EmbeddingQAFinetuneDataset): The dataset to evaluate.
        embed_model: The embedding model to use for evaluation.
        top_k (int): The number of top results to consider for evaluation.
        verbose (bool): Whether to print detailed evaluation information.
    Returns:
        list: A list of evaluation results containing hit status, MRR, retrieved IDs, expected ID, and query ID.
    """
    
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, show_progress=True
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc
        if is_hit:
            rank = retrieved_ids.index(expected_id) + 1
            mrr = 1 / rank
        else:
            mrr = 0
        eval_result = {
            "is_hit": is_hit,
            "mrr": mrr,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return pd.DataFrame(eval_results)


def plot_embeddings(pca, query_id, projected, query_embedding, expected_embedding, list_retrieved_embedding, save=True, filename="embeddings_pca.png"):
    query_dot = pca.transform([query_embedding])
    expected_dot = pca.transform([expected_embedding])
    retrieved_dot = pca.transform(list_retrieved_embedding)
    plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5)
    plt.scatter(query_dot[0, 0], query_dot[0, 1], color='green', label='Query', s=100)
    plt.scatter(expected_dot[0, 0], expected_dot[0, 1], color='red', label='Expected', s=100)
    for i, _ in enumerate(list_retrieved_embedding):
        plt.scatter(retrieved_dot[i, 0], retrieved_dot[i, 1], color='blue', label='Retrieved' if i == 0 else "", s=50)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title(f"PCA of Embeddings - query_id: {query_id}")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()


def dataframe_results(top_k, df_paraphrase_l12_results_it, finetuned=False):
    """ Create a DataFrame with the evaluation results for the embedding model.
    
    Returns:
        pd.DataFrame: A DataFrame containing the model name, top_k, MRR, and hit status.
    """
    df_results_embedding = pd.DataFrame()
    df_results_embedding["model"] = ["paraphrase-multilingual-MiniLM-L12-v2"]
    df_results_embedding["finetuned"] = [finetuned]
    df_results_embedding["top_k"] = [top_k]
    df_results_embedding["mrr"] = df_paraphrase_l12_results_it["mrr"].mean()
    df_results_embedding["is_hit"] = df_paraphrase_l12_results_it["is_hit"].mean()
    return df_results_embedding

def create_set_embeddings(it_dataset, embed_model):
    """
    Create embeddings for the dataset corpus.
    """
    set_embeddings = []
    for id_, text in tqdm(it_dataset.corpus.items()):
        embedding = embed_model.get_text_embedding(text)
        set_embeddings.append(embedding)
    return np.array(set_embeddings)


def get_embeddings(query_id, embedding_model, df_results, qa_dataset):
    " Get embeddings for a specific query from the results dataframe and dataset."
    _query = df_results[df_results["query"] == query_id]["query"].values[0]
    _expected = df_results[df_results["query"] == query_id]["expected"].values[0]
    _list_retrieved = df_results[df_results["query"] == query_id]["retrieved"].values[0]
    query_embedding = embedding_model.get_text_embedding(qa_dataset.queries[_query])
    expected_embedding = embedding_model.get_text_embedding(qa_dataset.corpus[_expected])
    list_retrieved_embedding = []
    for _, retrieved in enumerate(_list_retrieved):
        list_retrieved_embedding.append(embedding_model.get_text_embedding(qa_dataset.corpus[retrieved]))
    return query_embedding, expected_embedding, list_retrieved_embedding

def finetune_or_load_embedding_model(it_dataset, model_name, finetuned_filename, num_epochs):
    """Finetune the embedding model or load the finetuned model if it exists."""
    import os
    from loguru import logger
    from llama_index.finetuning import SentenceTransformersFinetuneEngine
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    if not os.path.exists(finetuned_filename):
        logger.info(f"Finetuning model {model_name} for {num_epochs} epochs...")
        finetune_engine = SentenceTransformersFinetuneEngine(
                                                        it_dataset,
                                                        model_id=model_name,
                                                        model_output_path=finetuned_filename,
                                                        val_dataset=it_dataset,
                                                        epochs=num_epochs
                                                        )
        finetune_engine.finetune()
        finetuned_embed_model = finetune_engine.get_finetuned_model()
        
    else:
        logger.info(f"Loading finetuned model from {finetuned_filename}...")
        finetuned_embed_model = HuggingFaceEmbedding(model_name=finetuned_filename)
    return finetuned_embed_model