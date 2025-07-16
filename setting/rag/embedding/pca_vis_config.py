class PCAConfig:
    top_k = 3
    dataset_name = "QA/gpt-35-turbo_dataset.json"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    num_pca_components = 2
    num_epochs = 2
    finetuned_filename = "finetuned-sentence-transformers/finetuned-paraphrase-multilingual-MiniLM-L12-v2"
