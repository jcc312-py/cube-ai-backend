#!/usr/bin/env python3
"""
Test script for database functionality
Run this to verify the database is working correctly
"""

import asyncio
from database import SafeDatabaseIntegration

async def test_database():
    print("🧪 Testing database functionality...")
    
    try:
        # Initialize database
        db = SafeDatabaseIntegration()
        print("✅ Database initialized successfully")
        
        # Test conversation creation
        conversation_id = await db.start_conversation("Test Conversation")
        print(f"✅ Created conversation: {conversation_id}")
        
        # Test getting conversations
        conversations = await db.get_conversations()
        print(f"✅ Found {len(conversations)} conversations")
        
        # Test getting specific conversation
        conversation = await db.get_conversation(conversation_id)
        if conversation:
            print(f"✅ Retrieved conversation: {conversation.title}")
        else:
            print("❌ Failed to retrieve conversation")
        
        print("🎉 All database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_database())
    if success:
        print("🚀 Database is ready to use!")
    else:
        print("💥 Database setup needs attention") 