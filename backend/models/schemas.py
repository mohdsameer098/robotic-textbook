from pydantic import BaseModel
from typing import Optional, List

class ChatMessage(BaseModel):
    message: str
    selected_text: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[dict] = []

class EmbedRequest(BaseModel):
    content: str
    metadata: dict