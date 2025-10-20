import os
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import Document

# ======================================================
# 🔧 CONFIGURATION
# ======================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "sk-yourkey"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_DIR = os.path.abspath("./data")  # Directory containing PDFs
INDEX_NAME = "finance_reports"

# ======================================================
# 🧱 STEP 1: CREATE DOCUMENT STORE
# ======================================================
document_store = InMemoryDocumentStore()  # No embedding_dim needed in v2.2

# ======================================================
# 🧾 STEP 2: LOAD & CLEAN PDF DOCUMENTS
# ======================================================
pdf_converter = PyPDFToDocument()
documents = []

print(f"📂 Reading PDFs from: {DATA_DIR}")

for file in os.listdir(DATA_DIR):
    if file.lower().endswith(".pdf"):
        path = os.path.join(DATA_DIR, file)
        if os.path.exists(path):
            try:
                result = pdf_converter.run(sources=[path])  # ✅ Haystack v2.2 syntax
                documents.extend(result["documents"])
                print(f"✅ Loaded: {file}")
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
        else:
            print(f"⚠️ File not found: {path}")

print(f"📄 Total documents loaded: {len(documents)}")

# Clean and split documents
cleaner = DocumentCleaner()
splitter = DocumentSplitter(split_by="word", split_length=200, split_overlap=50)

cleaned_result = cleaner.run(documents)
split_result = splitter.run(cleaned_result["documents"])

cleaned_docs = cleaned_result["documents"]
split_docs = split_result["documents"]


print(f"✂️ Split into {len(split_docs)} document chunks.")

# ======================================================
# 🧠 STEP 3: EMBED & INDEX DOCUMENTS
# ======================================================
embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
writer = DocumentWriter(document_store=document_store)

index_pipeline = Pipeline()
index_pipeline.add_component("embedder", embedder)
index_pipeline.add_component("writer", writer)
index_pipeline.connect("embedder.documents", "writer.documents")

print("🔄 Indexing documents into InMemory store...")
index_pipeline.run({"embedder": {"documents": split_docs}})
print(f"✅ Indexed {len(split_docs)} document chunks into InMemory store.")

# ======================================================
# 🤖 STEP 4: BUILD RAG PIPELINE (Modern v2.2 Design)
# ======================================================
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder

# Components
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
query_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
prompt_template = """
You are a financial analyst. Use the provided context to answer the question clearly and concisely.

Context:
{{ documents }}

Question:
{{ query }}

Answer:
"""
prompt_builder = PromptBuilder(template=prompt_template)
generator = OpenAIGenerator(model="gpt-4o-mini")

# Assemble pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("query_embedder", query_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("generator", generator)

# Connect the flow
rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")



# ======================================================
# 💬 STEP 5: ASK FINANCIAL QUESTIONS
# ======================================================
questions = [
    "What was Tesla’s total revenue in Q2 2025?",
    "Summarize Apple’s key financial highlights for Q2 FY23.",
    "How did Tesla’s net profit change compared to last quarter?"
]

print("\n🚀 Starting Financial Q&A...\n")

for q in questions:
    print("*" * 60)
    print(f"❓Question: {q}")
    try:
        result = rag_pipeline.run({
            "query_embedder": {"text": q},      # ✅ FIXED (was texts)
            "prompt_builder": {"query": q}
        })
        print(f"💡Answer: {result['generator']['replies'][0]}\n")
    except Exception as e:
        print(f"⚠️ Error during query: {e}")
