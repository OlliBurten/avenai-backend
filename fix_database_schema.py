#!/usr/bin/env python3
"""
Database Schema Migration Script
Fixes ChatSession and ChatMessage tables to match updated models
"""

import os
import sys
from sqlalchemy import text
from database import engine

def migrate_database():
    """Migrate database schema to match updated models"""
    print("🔧 Starting database migration...")
    
    try:
        with engine.connect() as conn:
            # Fix ChatSession table
            print("📝 Updating ChatSession table...")
            
            # Add title column if it doesn't exist
            conn.execute(text("""
                ALTER TABLE chat_sessions 
                ADD COLUMN IF NOT EXISTS title VARCHAR(200)
            """))
            
            # Add created_by column if it doesn't exist
            conn.execute(text("""
                ALTER TABLE chat_sessions 
                ADD COLUMN IF NOT EXISTS created_by VARCHAR(255)
            """))
            
            # Add updated_at column if it doesn't exist
            conn.execute(text("""
                ALTER TABLE chat_sessions 
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE
            """))
            
            # Remove old columns if they exist
            try:
                conn.execute(text("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS user_id"))
                conn.execute(text("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS session_token"))
                conn.execute(text("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS last_activity"))
            except Exception as e:
                print(f"⚠️  Warning removing old columns: {e}")
            
            # Fix ChatMessage table
            print("📝 Updating ChatMessage table...")
            
            # Add role column if it doesn't exist
            conn.execute(text("""
                ALTER TABLE chat_messages 
                ADD COLUMN IF NOT EXISTS role VARCHAR(20)
            """))
            
            # Add timestamp column if it doesn't exist
            conn.execute(text("""
                ALTER TABLE chat_messages 
                ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP WITH TIME ZONE
            """))
            
            # Add document_context column if it doesn't exist
            conn.execute(text("""
                ALTER TABLE chat_messages 
                ADD COLUMN IF NOT EXISTS document_context TEXT
            """))
            
            # Remove old columns if they exist
            try:
                conn.execute(text("ALTER TABLE chat_messages DROP COLUMN IF EXISTS message_type"))
                conn.execute(text("ALTER TABLE chat_messages DROP COLUMN IF EXISTS created_at"))
            except Exception as e:
                print(f"⚠️  Warning removing old columns: {e}")
            
            # Update existing data if needed
            print("🔄 Updating existing data...")
            
            # Set default values for new columns
            conn.execute(text("""
                UPDATE chat_sessions 
                SET title = 'API Support Chat' 
                WHERE title IS NULL
            """))
            
            conn.execute(text("""
                UPDATE chat_sessions 
                SET created_by = 'system' 
                WHERE created_by IS NULL
            """))
            
            conn.execute(text("""
                UPDATE chat_sessions 
                SET updated_at = created_at 
                WHERE updated_at IS NULL
            """))
            
            conn.execute(text("""
                UPDATE chat_messages 
                SET role = 'user' 
                WHERE role IS NULL
            """))
            
            conn.execute(text("""
                UPDATE chat_messages 
                SET timestamp = NOW() 
                WHERE timestamp IS NULL
            """))
            
            # Commit changes
            conn.commit()
            
            print("✅ Database migration completed successfully!")
            
            # Show final schema
            print("\n📊 Final table schemas:")
            
            # ChatSession schema
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'chat_sessions' 
                ORDER BY ordinal_position
            """))
            
            print("\nChatSession table:")
            for row in result:
                print(f"  - {row[0]}: {row[1]} ({'NULL' if row[2] == 'YES' else 'NOT NULL'})")
            
            # ChatMessage schema
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'chat_messages' 
                ORDER BY ordinal_position
            """))
            
            print("\nChatMessage table:")
            for row in result:
                print(f"  - {row[0]}: {row[1]} ({'NULL' if row[2] == 'YES' else 'NOT NULL'})")
                
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Avenai Database Migration Tool")
    print("=" * 40)
    
    success = migrate_database()
    
    if success:
        print("\n🎉 Migration completed! Your AI chat should now work.")
        print("💡 Try creating a chat session again.")
    else:
        print("\n💥 Migration failed. Check the error messages above.")
        sys.exit(1)
