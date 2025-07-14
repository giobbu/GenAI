from llama_index.finetuning import EmbeddingQAFinetuneDataset
from loguru import logger
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_cleaned_qa_dataset(dataset_name):
    """
    Load the dataset and clean it by removing empty entries.
    """
    logger.info(f"Loading dataset from {dataset_name}")
    it_dataset = EmbeddingQAFinetuneDataset.from_json(dataset_name)
    keys_to_delete = [key for key in it_dataset.corpus.keys() if it_dataset.corpus[key] == '']
    for key in keys_to_delete:
        del it_dataset.corpus[key]
    return it_dataset

def evaluate(dataset, embed_model, top_k=5, verbose=False):
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


def plot_embeddings(pca, projected, query_embedding, expected_embedding, retrieved_embedding):

    query_dot = pca.transform([query_embedding])
    expected_dot = pca.transform([expected_embedding])
    retrieved_dot = pca.transform([retrieved_embedding])

    plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5)
    plt.scatter(query_dot[0, 0], query_dot[0, 1], color='green', label='Query', s=100)
    plt.scatter(expected_dot[0, 0], expected_dot[0, 1], color='red', label='Expected', s=100)
    plt.scatter(retrieved_dot[0, 0], retrieved_dot[0, 1], color='blue', label='Retrieved', s=100)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title("Embeddings with Expected DIFFERENT from Retrieved")
    plt.legend()
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