import os
import sys
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))
load_dotenv(os.path.join(script_dir, ",env"))
load_dotenv()

# Normalize Google API key environment variables
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

if not os.environ.get("GOOGLE_API_KEY"):
    print("\n" + "=" * 60)
    print("⚠️  MISSING GOOGLE_API_KEY")
    print("To build the knowledge base for free with Google Gemini:")
    print("1. Get a free API key at: https://aistudio.google.com")
    print("2. Create a .env file in this directory with:")
    print("   GOOGLE_API_KEY=your_gemini_api_key_here")
    print("=" * 60 + "\n")
    sys.exit(1)

# 1. Load the document
from langchain_community.document_loaders import PyPDFLoader

base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, "vpn_user_guide.pdf")
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Missing {pdf_path}. Ensure the document is in the directory.")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split the document into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Create embeddings and build the FAISS vector store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Save the vector store to disk
faiss_dir = os.path.join(base_dir, "faiss_index")
vectorstore.save_local(faiss_dir)
print(f"Knowledge base created successfully from {len(documents)} pages, {len(chunks)} chunks at {faiss_dir}.")
