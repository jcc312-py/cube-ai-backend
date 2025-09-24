"""
Database module for multi-agent system
SAFE: This runs alongside existing agent communication without interfering
"""

import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
import uuid

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    workflows = relationship("Workflow", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    from_agent = Column(String, nullable=False)
    to_agent = Column(String, nullable=False)
    message_type = Column(String, nullable=False)  # task, data, request, response, error, status
    content = Column(Text, nullable=False)
    message_metadata = Column(JSON, default={})
    timestamp = Column(DateTime, default=datetime.utcnow)
    retry_count = Column(Integer, default=0)
    parent_message_id = Column(String, nullable=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

class Workflow(Base):
    __tablename__ = "workflows"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    agent_config = Column(JSON, nullable=False)  # Agent configurations
    status = Column(String, default="completed")  # running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="workflows")

class AgentMemory(Base):
    __tablename__ = "agent_memories"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    agent_id = Column(String, nullable=False)
    memory_type = Column(String, nullable=False)  # short_term, long_term, context
    content = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# =============================================================================
# DATABASE SERVICE
# =============================================================================

class DatabaseService:
    def __init__(self, db_url: str = "sqlite:///./agent_system.db"):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self):
        return self.SessionLocal()
    
    # Conversation Management
    def create_conversation(self, title: str) -> Conversation:
        with self.get_session() as session:
            conversation = Conversation(title=title)
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return conversation
    
    def get_conversations(self, limit: int = 50) -> List[Conversation]:
        with self.get_session() as session:
            conversations = session.query(Conversation).filter(
                Conversation.is_active == True
            ).order_by(Conversation.updated_at.desc()).limit(limit).all()
            
            # Convert to dictionaries to avoid session issues
            result = []
            for conv in conversations:
                # Get message count
                message_count = session.query(Message).filter(
                    Message.conversation_id == conv.id
                ).count()
                
                result.append({
                    'id': conv.id,
                    'title': conv.title,
                    'created_at': conv.created_at,
                    'updated_at': conv.updated_at,
                    'is_active': conv.is_active,
                    'message_count': message_count
                })
            return result
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        with self.get_session() as session:
            return session.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
    
    def delete_conversation(self, conversation_id: str) -> bool:
        with self.get_session() as session:
            conversation = session.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            if conversation:
                conversation.is_active = False
                session.commit()
                return True
            return False
    
    # Message Management
    def save_message(self, conversation_id: str, message_data: Dict[str, Any]) -> Message:
        with self.get_session() as session:
            message = Message(
                conversation_id=conversation_id,
                from_agent=message_data["from_agent"],
                to_agent=message_data["to_agent"],
                message_type=message_data["message_type"],
                content=message_data["content"],
                message_metadata=message_data.get("metadata", {}),
                timestamp=message_data.get("timestamp", datetime.utcnow()),
                retry_count=message_data.get("retry_count", 0),
                parent_message_id=message_data.get("parent_message_id")
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            return message
    
    def get_conversation_messages(self, conversation_id: str, limit: int = 100) -> List[Message]:
        with self.get_session() as session:
            return session.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.timestamp.asc()).limit(limit).all()
    
    # Agent Memory Management
    def save_agent_memory(self, conversation_id: str, agent_id: str, 
                               memory_type: str, content: Dict[str, Any]) -> AgentMemory:
        with self.get_session() as session:
            # Update existing memory or create new
            existing = session.query(AgentMemory).filter(
                AgentMemory.conversation_id == conversation_id,
                AgentMemory.agent_id == agent_id,
                AgentMemory.memory_type == memory_type
            ).first()
            
            if existing:
                existing.content = content
                existing.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(existing)
                return existing
            else:
                memory = AgentMemory(
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    memory_type=memory_type,
                    content=content
                )
                session.add(memory)
                session.commit()
                session.refresh(memory)
                return memory
    
    def get_agent_memory(self, conversation_id: str, agent_id: str, 
                              memory_type: str) -> Optional[AgentMemory]:
        with self.get_session() as session:
            return session.query(AgentMemory).filter(
                AgentMemory.conversation_id == conversation_id,
                AgentMemory.agent_id == agent_id,
                AgentMemory.memory_type == memory_type
            ).first()

# =============================================================================
# CONVERSATION LOGGER (SAFE INTEGRATION)
# =============================================================================

class ConversationLogger:
    """
    SAFE: This runs alongside the existing system without interfering
    It just logs what's happening for later retrieval
    """
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.current_conversation_id: Optional[str] = None
    
    async def start_conversation(self, title: str) -> str:
        """Start a new conversation - doesn't affect agent communication"""
        conversation = self.db_service.create_conversation(title)
        self.current_conversation_id = conversation.id
        return conversation.id
    
    async def log_message(self, message):
        """Log a message to database - doesn't affect agent processing"""
        if self.current_conversation_id:
            try:
                self.db_service.save_message(
                    self.current_conversation_id, 
                    {
                        "from_agent": message.from_agent,
                        "to_agent": message.to_agent,
                        "message_type": message.message_type.value,
                        "content": message.content,
                        "metadata": message.metadata,
                        "timestamp": message.timestamp,
                        "retry_count": message.retry_count,
                        "parent_message_id": message.parent_message_id
                    }
                )
            except Exception as e:
                print(f"⚠️ Database logging failed (non-critical): {e}")
    
    async def log_agent_memory(self, agent_id: str, memory_data: Dict[str, Any]):
        """Log agent memory - doesn't affect agent operation"""
        if self.current_conversation_id:
            try:
                self.db_service.save_agent_memory(
                    self.current_conversation_id,
                    agent_id,
                    "short_term",
                    memory_data
                )
            except Exception as e:
                print(f"⚠️ Memory logging failed (non-critical): {e}")

# =============================================================================
# SAFE DATABASE INTEGRATION
# =============================================================================

class SafeDatabaseIntegration:
    """
    SAFE: This class provides database functionality without breaking existing flow
    """
    
    def __init__(self, db_url: str = "sqlite:///./agent_system.db"):
        self.db_service = DatabaseService(db_url)
        self.logger = ConversationLogger(self.db_service)
        self.enabled = True  # Can be disabled if needed
    
    def attach_to_message_bus(self, message_bus, conversation_id: Optional[str] = None):
        """Safely attach logger to existing message bus"""
        message_bus.logger = self.logger
        if conversation_id:
            self.logger.current_conversation_id = conversation_id
    
    async def start_conversation(self, title: str) -> str:
        """Start new conversation - doesn't affect agents"""
        if self.enabled:
            return await self.logger.start_conversation(title)
        return None
    
    async def get_conversations(self):
        """Get conversation history - read-only operation"""
        if self.enabled:
            return self.db_service.get_conversations()
        return []
    
    async def get_conversation(self, conversation_id: str):
        """Get specific conversation - read-only"""
        if self.enabled:
            return self.db_service.get_conversation(conversation_id)
        return None
    
    async def add_message_to_conversation(self, conversation_id: str, from_agent: str, 
                                        to_agent: str, message_type: str, content: str, 
                                        metadata: Dict[str, Any] = None):
        """Add message to conversation"""
        if self.enabled:
            message_data = {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "message_type": message_type,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow()
            }
            return self.db_service.save_message(conversation_id, message_data)
        return None
    
    async def delete_conversation(self, conversation_id: str):
        """Delete conversation - doesn't affect current agents"""
        if self.enabled:
            return self.db_service.delete_conversation(conversation_id)
        return False
    
    def disable(self):
        """Disable database functionality if needed"""
        self.enabled = False
        print("🔴 Database integration disabled")
    
    def enable(self):
        """Enable database functionality"""
        self.enabled = True
        print("🟢 Database integration enabled")

# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================

class ConversationRequest(BaseModel):
    title: str

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

class ChatRequestWithConversation(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    code_history: List[str] = []
    error_history: List[str] = [] 