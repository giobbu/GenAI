from utils import StructuredOutput, generate_response
from llama_index.llms.ollama import Ollama

def run_synthetizer(fmt_qa_prompt, params):
    """Run the LLM to generate a response based on the formatted prompt."""
    llm = Ollama(
                model=params.llm_model_name,
                request_timeout=params.request_timeout,
                context_window=params.context_window,
                json_mode=params.json_mode
            )
    structured_llm = llm.as_structured_llm(StructuredOutput)
    response = generate_response(fmt_qa_prompt, structured_llm)
    return response