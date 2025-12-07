import os
import aiosqlite
from dotenv import load_dotenv

load_dotenv()

class DatabaseService:
    def __init__(self):
        self.db_path = "chatbot.db"
    
    async def initialize_tables(self):
        """Create necessary tables if they don't exist"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    selected_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    qdrant_id TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.commit()
        print("✅ Database tables initialized")
    
    async def save_chat(self, user_message: str, bot_response: str, selected_text: str = None):
        """Save chat interaction to database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO chat_history (user_message, bot_response, selected_text)
                VALUES (?, ?, ?)
            """, (user_message, bot_response, selected_text))
            await db.commit()