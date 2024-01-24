"""
Basic RAG involving loading and vectorizing a paper, 
and then using a local Mistral LLM along with LlamaIndex to
query and use the created vectorDB to answer questions.
"""
#!/usr/bin/env python

import time
# download_loader allows for loading of PDFs
from llama_index import VectorStoreIndex, ServiceContext, download_loader
from llama_index.llms import LlamaCPP

# These utils help format model input
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

#Initialize custom loader
PDFReader = download_loader("PDFReader")
loader = PDFReader()

# Read PDF File
documents = loader.load_data(file="./1804.00792.pdf")


# We can see that it splits it into each individual page.
print(f"Number of Docs: {len(documents)}")
# We can print one of these pages
print(documents[0])

MODEL_PATH = "./models/mistral-7b-2.20bpw.gguf"

llm = LlamaCPP(
    # Can also set the path to a pre downloaded model instead
    model_path=MODEL_PATH,
    temperature=0.1,
    max_new_tokens=512,
    # llama2 has a context window of 4096 tokens
    context_window=4096,
    generate_kwargs={},
    # Use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False,
)

# Setting up the ServiceContext with the language model and embedding model
EMBED_MODEL = "local:BAAI/bge-small-en-v1.5"
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=EMBED_MODEL
)

# Creating the VectorStoreIndex for document handling
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Converting the index to a query engine for retrieval
query_engine = index.as_query_engine()

def query_and_display(question):
    """
    Simple function to query the LLM and time the output generation.
    """
    tic = time.time()
    response = query_engine.query(question)
    print(response)
    print(f"Took {time.time()-tic}s")

query_and_display("Who wrote the Poison Frogs paper?")
query_and_display("What is Watermarking in the context of the paper?")
