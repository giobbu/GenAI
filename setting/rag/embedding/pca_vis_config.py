class PCAConfig:
    top_k = 3
    dataset_name = "QA/codice/gpt-35-turbo_temperature_1_dataset.json"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    num_pca_components = 2
    num_epochs = 2
    finetuned_filename = "finetuned-sentence-transformers/codice/finetuned-paraphrase-multilingual-MiniLM-L12-v2_gpt-35-turbo_temperature_1_dataset"
