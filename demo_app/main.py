import streamlit as st
from llama_index.llms.ollama import Ollama
from setting import configLLM
from utils import StructuredOutput, create_document, generate_response
import json
import time

def main():
    st.title("Q&A Demo App")

    # Input query from user
    user_query = st.text_input("Enter your question:", value="What is Apple?")

    if st.button("Get Answer"):
        with st.spinner("Generating response..."):

            params = configLLM()

            llm = Ollama(
                model=params.model_name,
                request_timeout=params.request_timeout,
                context_window=params.context_window,
                json_mode=params.json_mode
            )

            # Create structured LLM
            structured_llm = llm.as_structured_llm(StructuredOutput)

            # Generate and display response
            start_time = time.time()
            text_dict = generate_response(query=user_query, llm=structured_llm)
            latency = time.time() - start_time
            text_dict['latency'] = latency
            st.subheader("Structured Response")
            st.json(text_dict)

            # Convert to downloadable string
            json_str = json.dumps(text_dict, indent=2)

            # Save to .docx
            filename = "llm_response.docx"
            create_document(text_dict, filename=filename)

            # Open and prepare for download
            with open(filename, "rb") as f:
                docx_bytes = f.read()

            st.download_button(
                label="📄 Download Word Document",
                data=docx_bytes,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )


if __name__ == "__main__":
    main()
