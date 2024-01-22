# localRAG
Build a local RAG system using Mistral and LlamaIndex


## LlamaIndex
Data framework to build LLM apps by providing data connectors to ingest data sources - think PDFs, docs, SQL, APIs; Ways to structure data - graph and indices; Advanced retrieval and query interface over the data.

## Llama.cpp
For the local LLM, I'll use Llama.cpp since its better optimized for running LLMs with less available RAM, it also has support for CUDA. 

### Installation
`llama-cpp-python` not only offers Python bindings for `llama.cpp` but it also builds from source directly, making it easy to use. 

By default, it will build for CPU on Linux and Windows systems, so ensure you run the following to build for CUDA enabled GPUS: 

```CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python```