from dotenv import load_dotenv
import os

load_dotenv()

print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))
print("GROQ_FALLBACK_API_KEY set:", bool(os.getenv("GROQ_FALLBACK_API_KEY")))
print("GEMINI_API_KEY set:", bool(os.getenv("GEMINI_API_KEY")))
print("QDRANT_URL set:", bool(os.getenv("QDRANT_URL")))
print("QDRANT_API_KEY set:", bool(os.getenv("QDRANT_API_KEY")))
