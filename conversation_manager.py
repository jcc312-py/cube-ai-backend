"""
Unified Conversation Management System
Handles conversations for both offline and online agents with proper organization
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading

# =============================================================================
# DATA MODELS
# =============================================================================

class ConversationType(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    MANUAL = "manual"
    WORKFLOW = "workflow"

class MessageType(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    ERROR = "error"
    STATUS = "status"

@dataclass
class Conversation:
    id: str
    title: str
    type: ConversationType
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Message:
    id: str
    conversation_id: str
    type: MessageType
    content: str
    agent_id: Optional[str] = None
    agent_role: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PromptTemplate:
    id: str
    name: str
    description: str
    content: str
    category: str
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    usage_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.usage_count is None:
            self.usage_count = 0

# =============================================================================
# CONVERSATION MANAGER
# =============================================================================

class ConversationManager:
    """Unified conversation management for offline and online agents"""
    
    def __init__(self, data_dir: str = "conversation_data"):
        try:
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(exist_ok=True)
            
            # Initialize database
            self.db_path = self.data_dir / "conversations.db"
            self._init_database()
            
            # Thread safety
            self._lock = threading.Lock()
            
            # Load default prompt templates (commented out for now to avoid errors)
            # self._load_default_templates()
            
            print(f"✅ Conversation manager initialized with database at: {self.db_path}")
        except Exception as e:
            print(f"❌ Failed to initialize conversation manager: {e}")
            # Create a fallback in-memory manager
            self.data_dir = None
            self.db_path = None
            self._lock = threading.Lock()
            self._conversations = {}
            self._messages = {}
            self._templates = {}
    
    def load_default_templates(self):
        """Load default prompt templates manually"""
        self._load_default_templates()
    
    def _init_database(self):
        """Initialize SQLite database for conversations"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_id TEXT,
                    agent_role TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_templates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    tags TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_type ON conversations (type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages (timestamp)
            """)
    
    def _load_default_templates(self):
        """Load default prompt templates"""
        default_templates = [
            PromptTemplate(
                id="coding_task",
                name="Coding Task",
                description="Template for coding tasks and development work",
                content="""I need help with a coding task. Please provide:

1. **Task Description**: {task_description}
2. **Technology Stack**: {technology_stack}
3. **Requirements**: {requirements}
4. **Expected Output**: {expected_output}

Please write clean, well-documented code and include tests if applicable.""",
                category="Development",
                tags=["coding", "development", "programming"]
            ),
            PromptTemplate(
                id="bug_fix",
                name="Bug Fix",
                description="Template for debugging and fixing issues",
                content="""I'm experiencing a bug in my code. Please help me fix it:

1. **Problem Description**: {problem_description}
2. **Error Messages**: {error_messages}
3. **Code Context**: {code_context}
4. **Expected Behavior**: {expected_behavior}
5. **Actual Behavior**: {actual_behavior}

Please analyze the issue and provide a solution with explanations.""",
                category="Debugging",
                tags=["bug", "debug", "fix", "error"]
            ),
            PromptTemplate(
                id="code_review",
                name="Code Review",
                description="Template for code review and improvement suggestions",
                content="""Please review the following code and provide feedback:

1. **Code to Review**: {code_to_review}
2. **Review Focus**: {review_focus}
3. **Specific Concerns**: {specific_concerns}

Please provide:
- Code quality assessment
- Performance improvements
- Security considerations
- Best practices recommendations
- Refactoring suggestions""",
                category="Code Review",
                tags=["review", "quality", "improvement", "refactor"]
            ),
            PromptTemplate(
                id="documentation",
                name="Documentation",
                description="Template for creating documentation",
                content="""I need help creating documentation for:

1. **Project/Feature**: {project_name}
2. **Documentation Type**: {doc_type}
3. **Target Audience**: {target_audience}
4. **Key Points to Cover**: {key_points}

Please create comprehensive, clear documentation that is easy to understand.""",
                category="Documentation",
                tags=["docs", "documentation", "writing", "guide"]
            ),
            PromptTemplate(
                id="testing",
                name="Testing",
                description="Template for creating tests",
                content="""I need help creating tests for my code:

1. **Code to Test**: {code_to_test}
2. **Test Type**: {test_type}
3. **Test Framework**: {test_framework}
4. **Coverage Requirements**: {coverage_requirements}

Please create comprehensive tests including unit tests, integration tests, and edge cases.""",
                category="Testing",
                tags=["testing", "test", "coverage", "quality"]
            )
        ]
        
        for template in default_templates:
            self.save_prompt_template(template)
    
    # =============================================================================
    # CONVERSATION MANAGEMENT
    # =============================================================================
    
    def create_conversation(self, title: str, conversation_type: ConversationType, 
                          metadata: Dict[str, Any] = None) -> Conversation:
        """Create a new conversation"""
        with self._lock:
            conversation_id = str(uuid.uuid4())
            now = datetime.now()
            
            conversation = Conversation(
                id=conversation_id,
                title=title,
                type=conversation_type,
                created_at=now,
                updated_at=now,
                metadata=metadata or {}
            )
            
            if not self.db_path:
                # Fallback to in-memory storage
                self._conversations[conversation_id] = conversation
                return conversation
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO conversations (id, title, type, created_at, updated_at, is_active, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation.id,
                    conversation.title,
                    conversation.type.value,
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                    conversation.is_active,
                    json.dumps(conversation.metadata)
                ))
            
            return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, title, type, created_at, updated_at, is_active, metadata
                FROM conversations WHERE id = ?
            """, (conversation_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return Conversation(
                id=row[0],
                title=row[1],
                type=ConversationType(row[2]),
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4]),
                is_active=bool(row[5]),
                metadata=json.loads(row[6]) if row[6] else {}
            )
    
    def list_conversations(self, conversation_type: Optional[ConversationType] = None,
                          limit: int = 50, offset: int = 0) -> List[Conversation]:
        """List conversations with optional filtering"""
        if not self.db_path:
            # Fallback to in-memory storage
            conversations = list(self._conversations.values())
            if conversation_type:
                conversations = [c for c in conversations if c.type == conversation_type]
            return conversations[offset:offset + limit]
        
        with sqlite3.connect(self.db_path) as conn:
            if conversation_type:
                cursor = conn.execute("""
                    SELECT id, title, type, created_at, updated_at, is_active, metadata
                    FROM conversations 
                    WHERE type = ? AND is_active = 1
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """, (conversation_type.value, limit, offset))
            else:
                cursor = conn.execute("""
                    SELECT id, title, type, created_at, updated_at, is_active, metadata
                    FROM conversations 
                    WHERE is_active = 1
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append(Conversation(
                    id=row[0],
                    title=row[1],
                    type=ConversationType(row[2]),
                    created_at=datetime.fromisoformat(row[3]),
                    updated_at=datetime.fromisoformat(row[4]),
                    is_active=bool(row[5]),
                    metadata=json.loads(row[6]) if row[6] else {}
                ))
            
            return conversations
    
    def update_conversation(self, conversation_id: str, **kwargs) -> bool:
        """Update conversation properties"""
        with self._lock:
            allowed_fields = ['title', 'is_active', 'metadata']
            updates = []
            values = []
            
            for field, value in kwargs.items():
                if field in allowed_fields:
                    if field == 'metadata':
                        updates.append(f"{field} = ?")
                        values.append(json.dumps(value))
                    else:
                        updates.append(f"{field} = ?")
                        values.append(value)
            
            if not updates:
                return False
            
            updates.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(conversation_id)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"""
                    UPDATE conversations 
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, values)
                
                return cursor.rowcount > 0
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Delete messages first
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                
                # Delete conversation
                cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
                return cursor.rowcount > 0
    
    # =============================================================================
    # MESSAGE MANAGEMENT
    # =============================================================================
    
    def add_message(self, conversation_id: str, message_type: MessageType, 
                   content: str, agent_id: Optional[str] = None,
                   agent_role: Optional[str] = None, metadata: Dict[str, Any] = None) -> Message:
        """Add a message to a conversation"""
        with self._lock:
            message_id = str(uuid.uuid4())
            now = datetime.now()
            
            message = Message(
                id=message_id,
                conversation_id=conversation_id,
                type=message_type,
                content=content,
                agent_id=agent_id,
                agent_role=agent_role,
                timestamp=now,
                metadata=metadata or {}
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO messages (id, conversation_id, type, content, agent_id, agent_role, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    message.conversation_id,
                    message.type.value,
                    message.content,
                    message.agent_id,
                    message.agent_role,
                    message.timestamp.isoformat(),
                    json.dumps(message.metadata)
                ))
                
                # Update conversation timestamp
                conn.execute("""
                    UPDATE conversations 
                    SET updated_at = ?
                    WHERE id = ?
                """, (now.isoformat(), conversation_id))
            
            return message
    
    def get_messages(self, conversation_id: str, limit: int = 100, 
                    offset: int = 0) -> List[Message]:
        """Get messages for a conversation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, conversation_id, type, content, agent_id, agent_role, timestamp, metadata
                FROM messages 
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
            """, (conversation_id, limit, offset))
            
            messages = []
            for row in cursor.fetchall():
                messages.append(Message(
                    id=row[0],
                    conversation_id=row[1],
                    type=MessageType(row[2]),
                    content=row[3],
                    agent_id=row[4],
                    agent_role=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    metadata=json.loads(row[7]) if row[7] else {}
                ))
            
            return messages
    
    def get_conversation_with_messages(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation with all its messages"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        messages = self.get_messages(conversation_id)
        
        return {
            "conversation": asdict(conversation),
            "messages": [asdict(msg) for msg in messages],
            "message_count": len(messages)
        }
    
    # =============================================================================
    # PROMPT TEMPLATE MANAGEMENT
    # =============================================================================
    
    def save_prompt_template(self, template: PromptTemplate) -> bool:
        """Save or update a prompt template"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Check if template exists
                cursor = conn.execute("SELECT id FROM prompt_templates WHERE id = ?", (template.id,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing template
                    conn.execute("""
                        UPDATE prompt_templates 
                        SET name = ?, description = ?, content = ?, category = ?, 
                            tags = ?, updated_at = ?, usage_count = ?
                        WHERE id = ?
                    """, (
                        template.name,
                        template.description,
                        template.content,
                        template.category,
                        json.dumps(template.tags),
                        template.updated_at.isoformat(),
                        template.usage_count,
                        template.id
                    ))
                else:
                    # Insert new template
                    conn.execute("""
                        INSERT INTO prompt_templates (id, name, description, content, category, tags, created_at, updated_at, usage_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        template.id,
                        template.name,
                        template.description,
                        template.content,
                        template.category,
                        json.dumps(template.tags),
                        template.created_at.isoformat(),
                        template.updated_at.isoformat(),
                        template.usage_count
                    ))
                
                return True
    
    def get_prompt_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a prompt template by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, name, description, content, category, tags, created_at, updated_at, usage_count
                FROM prompt_templates WHERE id = ?
            """, (template_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return PromptTemplate(
                id=row[0],
                name=row[1],
                description=row[2],
                content=row[3],
                category=row[4],
                tags=json.loads(row[5]) if row[5] else [],
                created_at=datetime.fromisoformat(row[6]),
                updated_at=datetime.fromisoformat(row[7]),
                usage_count=row[8]
            )
    
    def list_prompt_templates(self, category: Optional[str] = None) -> List[PromptTemplate]:
        """List prompt templates with optional category filtering"""
        with sqlite3.connect(self.db_path) as conn:
            if category:
                cursor = conn.execute("""
                    SELECT id, name, description, content, category, tags, created_at, updated_at, usage_count
                    FROM prompt_templates 
                    WHERE category = ?
                    ORDER BY usage_count DESC, name ASC
                """, (category,))
            else:
                cursor = conn.execute("""
                    SELECT id, name, description, content, category, tags, created_at, updated_at, usage_count
                    FROM prompt_templates 
                    ORDER BY usage_count DESC, name ASC
                """)
            
            templates = []
            for row in cursor.fetchall():
                templates.append(PromptTemplate(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    content=row[3],
                    category=row[4],
                    tags=json.loads(row[5]) if row[5] else [],
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    usage_count=row[8]
                ))
            
            return templates
    
    def increment_template_usage(self, template_id: str) -> bool:
        """Increment usage count for a template"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE prompt_templates 
                SET usage_count = usage_count + 1, updated_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), template_id))
            
            return cursor.rowcount > 0
    
    def delete_prompt_template(self, template_id: str) -> bool:
        """Delete a prompt template"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM prompt_templates WHERE id = ?", (template_id,))
            return cursor.rowcount > 0
    
    # =============================================================================
    # SEARCH AND ANALYTICS
    # =============================================================================
    
    def search_conversations(self, query: str, conversation_type: Optional[ConversationType] = None) -> List[Conversation]:
        """Search conversations by title and content"""
        with sqlite3.connect(self.db_path) as conn:
            if conversation_type:
                cursor = conn.execute("""
                    SELECT DISTINCT c.id, c.title, c.type, c.created_at, c.updated_at, c.is_active, c.metadata
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE c.type = ? AND c.is_active = 1 
                    AND (c.title LIKE ? OR m.content LIKE ?)
                    ORDER BY c.updated_at DESC
                """, (conversation_type.value, f"%{query}%", f"%{query}%"))
            else:
                cursor = conn.execute("""
                    SELECT DISTINCT c.id, c.title, c.type, c.created_at, c.updated_at, c.is_active, c.metadata
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE c.is_active = 1 
                    AND (c.title LIKE ? OR m.content LIKE ?)
                    ORDER BY c.updated_at DESC
                """, (f"%{query}%", f"%{query}%"))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append(Conversation(
                    id=row[0],
                    title=row[1],
                    type=ConversationType(row[2]),
                    created_at=datetime.fromisoformat(row[3]),
                    updated_at=datetime.fromisoformat(row[4]),
                    is_active=bool(row[5]),
                    metadata=json.loads(row[6]) if row[6] else {}
                ))
            
            return conversations
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total conversations
            cursor = conn.execute("SELECT COUNT(*) FROM conversations WHERE is_active = 1")
            total_conversations = cursor.fetchone()[0]
            
            # Conversations by type
            cursor = conn.execute("""
                SELECT type, COUNT(*) FROM conversations 
                WHERE is_active = 1 
                GROUP BY type
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Total messages
            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            
            # Recent activity (last 7 days)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM conversations 
                WHERE is_active = 1 AND updated_at >= datetime('now', '-7 days')
            """)
            recent_conversations = cursor.fetchone()[0]
            
            return {
                "total_conversations": total_conversations,
                "conversations_by_type": by_type,
                "total_messages": total_messages,
                "recent_conversations": recent_conversations
            }

# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Global conversation manager instance
conversation_manager = ConversationManager()
