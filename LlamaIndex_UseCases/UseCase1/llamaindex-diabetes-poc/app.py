from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import os

# Load OpenAI-compatible LLM (can be Azure/OpenAI/local endpoint too)
llm = OpenAI(model="gpt-4o-mini")  # You can change the model

# Load medical data
documents = SimpleDirectoryReader("data").load_data()

# Build vector index
index = VectorStoreIndex.from_documents(documents, llm=llm)
query_engine = index.as_query_engine()

print("✅ Diabetes Clinical Support System Ready")

while True:
    query = input("\nEnter patient symptoms/metrics (or 'exit'): ")
    if query.lower() == "exit":
        break
    
    response = query_engine.query(query)
    print("\n📌 Medical Guidance:")
    print(response)
