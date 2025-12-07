import os
import sys
import time
from pathlib import Path
from services.gemini_service import GeminiService
from services.qdrant_service import QdrantService
from services.database import DatabaseService
import asyncio

def read_markdown_files(docs_path: str):
    """Read all markdown files from the docs directory"""
    docs_dir = Path(docs_path)
    markdown_files = []
    
    for md_file in docs_dir.rglob("*.md"):
        if md_file.name != "README.md":
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                markdown_files.append({
                    "file_path": str(md_file),
                    "file_name": md_file.name,
                    "content": content
                })
    
    return markdown_files

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

async def embed_documents():
    """Main function to embed all documents"""
    print("🚀 Starting document embedding process...")
    
    # Initialize services
    gemini = GeminiService()
    qdrant = QdrantService()
    db = DatabaseService()
    
    # Initialize database and Qdrant collection
    await db.initialize_tables()
    qdrant.initialize_collection(vector_size=768)
    
    # Read markdown files from docs directory
    docs_path = "../docs"
    
    if not os.path.exists(docs_path):
        print(f"❌ Docs directory not found at: {docs_path}")
        print("Please adjust the path in embed_content.py")
        return
    
    markdown_files = read_markdown_files(docs_path)
    print(f"📚 Found {len(markdown_files)} markdown files")
    
    # Process each file
    total_chunks = 0
    request_count = 0
    
    for idx, doc in enumerate(markdown_files, 1):
        print(f"\n📄 Processing ({idx}/{len(markdown_files)}): {doc['file_name']}")
        
        # Split into chunks
        chunks = chunk_text(doc['content'])
        print(f"   📦 Created {len(chunks)} chunks")
        
        # Embed each chunk with rate limiting
        for chunk_idx, chunk in enumerate(chunks):
            try:
                # Rate limiting: 15 requests per minute for free tier
                if request_count > 0 and request_count % 10 == 0:
                    print(f"   ⏳ Rate limit: Waiting 60 seconds...")
                    time.sleep(60)
                
                # Generate embedding
                embedding = gemini.generate_embeddings(chunk)
                
                # Store in Qdrant
                metadata = {
                    "file_name": doc['file_name'],
                    "file_path": doc['file_path'],
                    "chunk_index": chunk_idx
                }
                
                qdrant.add_document(embedding, chunk, metadata)
                total_chunks += 1
                request_count += 1
                
                print(f"   ✅ Embedded chunk {chunk_idx + 1}/{len(chunks)}")
                
                # Small delay between requests
                time.sleep(2)
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"   ⏳ Rate limit hit! Waiting 60 seconds...")
                    time.sleep(60)
                    # Retry this chunk
                    try:
                        embedding = gemini.generate_embeddings(chunk)
                        metadata = {
                            "file_name": doc['file_name'],
                            "file_path": doc['file_path'],
                            "chunk_index": chunk_idx
                        }
                        qdrant.add_document(embedding, chunk, metadata)
                        total_chunks += 1
                        print(f"   ✅ Retry successful for chunk {chunk_idx + 1}")
                    except Exception as retry_error:
                        print(f"   ❌ Retry failed for chunk {chunk_idx}: {retry_error}")
                else:
                    print(f"   ❌ Error embedding chunk {chunk_idx}: {e}")
    
    print(f"\n🎉 Embedding complete! Total chunks embedded: {total_chunks}")

if __name__ == "__main__":
    asyncio.run(embed_documents())