# Experimentation

## Folder `datasets`

* different data sources. E.g., `data_source_1` and `data_source_2`.
* different chunking strategies. E.g., `dataset_chunking_1`
* different parsing strategies. E.g., `dataset_ocr_1`

## Module `prompt-engineering`

Set of template created. 
For each `template`, the template creation uses set of `examples` and `instructions`.

```python
f"""
Answer the {user_query}.
... {instruction} ...
... {example_1} ... {example_2}.
"""
```

and more advanced template, e.g., templates where `rag_context` from grounding the llm's response is given from a RAG.

```python
f"""
Given the {rag_context}.
Answer the {user_query}.
... {instruction} ...
... {example_1}.
"""
```

## Module `finetuning`

* `embedding` params:
    * `dim`
* `retrieving` strategies:
    * `sentence-window` params
    * `auto-merging` params
* `re-ranking`:
    * params: `top-k`, etc.
* `generative` strategies:
    * `lora`:
        * params
    * `qlora`:
        * params

## Module `optimization`

1. select optmization procedure (e.g., grid search, random search, bayesian optimization)
2. try difefrent configurations, `config`:
    * for `dataset` in `list_datasets`:
        * for `template` in `list_templates`:
            * for `finetuning_strategy` in `list_finetuning_strategies`:
                * for `model_params` in `list_model_params`:
                    * TRAIN & VALIDATE:
                        * config : (`dataset` - `template` - `finetuning-strategy` - `model_params`) [**instance config - model serving pipeline**];
                        * collect results experiments
3. select `best_config` and save to `artifacts-registry`  
4. deploy the solution moving to `serving` layer

