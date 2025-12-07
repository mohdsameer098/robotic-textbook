import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid

load_dotenv()

class QdrantService:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("⚠️ QDRANT_URL or QDRANT_API_KEY not set in .env file")
        
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        self.collection_name = "robotics_textbook"
        
    def initialize_collection(self, vector_size: int = 768):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
            print(f"✅ Collection '{self.collection_name}' already exists")
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Collection '{self.collection_name}' created")
    
    def add_document(self, embedding: list, text: str, metadata: dict):
        """Add a document to Qdrant"""
        point_id = str(uuid.uuid4())
        
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "text": text,
                "metadata": metadata
            }
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        return point_id
    
    def search_similar(self, query_embedding: list, limit: int = 5):
        """Search for similar documents"""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        results = []
        for hit in search_result:
            results.append({
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            })
        
        return results