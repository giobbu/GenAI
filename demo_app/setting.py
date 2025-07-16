class configLLM:
    model_name = "Almawave/velvet:latest"
    request_timeout = 240.0
    json_mode= True
    context_window = 8000
    additional_kwargs={"seed": 42,
                       "temperature": 0.1  # lower = more deterministic
                       }