import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_key_here":
            raise ValueError("⚠️ GEMINI_API_KEY not set in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = 'models/embedding-001'
    
    def generate_embeddings(self, text: str):
        """Generate embeddings for text using Gemini"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    def generate_query_embedding(self, query: str):
        """Generate embeddings for search query"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def generate_response(self, prompt: str, context: str = ""):
        """Generate response using Gemini with context"""
        full_prompt = f"""You are a helpful assistant for a Physical AI & Humanoid Robotics textbook.

Context from the book:
{context}

User Question: {prompt}

Please provide a clear, accurate answer based on the context provided. If the context doesn't contain enough information, say so politely."""

        response = self.model.generate_content(full_prompt)
        return response.text