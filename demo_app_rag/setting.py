class configRAG:
    llm_model_name = "mistral-small:22b"
    embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    similarity_top_k = 2
    chunk_size = 512
    chunk_overlap = 0 # overlap between chunks
    request_timeout = 240.0
    json_mode= True
    context_window = 8000
    additional_kwargs={"seed": 42,
                       "temperature": 0.1  # lower = more deterministic
                       }
    directory = r"doc/"
    required_exts = [".txt"]