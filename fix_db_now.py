#!/usr/bin/env python3
"""
Emergency Database Fix Script
Run this directly to fix the chat_sessions table schema
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection (use your Railway PostgreSQL URL)
DATABASE_URL = "postgresql://postgres:TLqDHVnPDUFqRZgaCiXbPPmtyskKvdrH@ballast.proxy.rlwy.net:16651/railway"

def fix_database():
    """Fix the database schema immediately"""
    print("🚨 EMERGENCY DATABASE FIX")
    print("=" * 40)
    
    try:
        # Connect to database
        print("🔌 Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        print("✅ Connected to database!")
        
        # Fix ChatSession table
        print("\n📝 Fixing ChatSession table...")
        
        # Add title column
        try:
            cur.execute("""
                ALTER TABLE chat_sessions 
                ADD COLUMN title VARCHAR(200)
            """)
            print("✅ Added 'title' column")
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️  'title' column already exists")
            else:
                print(f"⚠️  Error with title column: {e}")
        
        # Add created_by column
        try:
            cur.execute("""
                ALTER TABLE chat_sessions 
                ADD COLUMN created_by VARCHAR(255)
            """)
            print("✅ Added 'created_by' column")
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️  'created_by' column already exists")
            else:
                print(f"⚠️  Error with created_by column: {e}")
        
        # Add updated_at column
        try:
            cur.execute("""
                ALTER TABLE chat_sessions 
                ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE
            """)
            print("✅ Added 'updated_at' column")
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️  'updated_at' column already exists")
            else:
                print(f"⚠️  Error with updated_at column: {e}")
        
        # Fix ChatMessage table
        print("\n📝 Fixing ChatMessage table...")
        
        # Add role column
        try:
            cur.execute("""
                ALTER TABLE chat_messages 
                ADD COLUMN role VARCHAR(20)
            """)
            print("✅ Added 'role' column")
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️  'role' column already exists")
            else:
                print(f"⚠️  Error with role column: {e}")
        
        # Add timestamp column
        try:
            cur.execute("""
                ALTER TABLE chat_messages 
                ADD COLUMN timestamp TIMESTAMP WITH TIME ZONE
            """)
            print("✅ Added 'timestamp' column")
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️  'timestamp' column already exists")
            else:
                print(f"⚠️  Error with timestamp column: {e}")
        
        # Add document_context column
        try:
            cur.execute("""
                ALTER TABLE chat_messages 
                ADD COLUMN document_context TEXT
            """)
            print("✅ Added 'document_context' column")
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️  'document_context' column already exists")
            else:
                print(f"⚠️  Error with document_context column: {e}")
        
        # Set default values
        print("\n🔄 Setting default values...")
        
        cur.execute("""
            UPDATE chat_sessions 
            SET title = 'API Support Chat' 
            WHERE title IS NULL
        """)
        print("✅ Set default titles")
        
        cur.execute("""
            UPDATE chat_sessions 
            SET created_by = 'system' 
            WHERE created_by IS NULL
        """)
        print("✅ Set default created_by")
        
        cur.execute("""
            UPDATE chat_sessions 
            SET updated_at = created_at 
            WHERE updated_at IS NULL
        """)
        print("✅ Set default updated_at")
        
        # Commit changes
        conn.commit()
        print("\n✅ All changes committed!")
        
        # Show final schema
        print("\n📊 Final ChatSession table schema:")
        cur.execute("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'chat_sessions' 
            ORDER BY ordinal_position
        """)
        
        for row in cur.fetchall():
            print(f"  - {row[0]}: {row[1]} ({'NULL' if row[2] == 'YES' else 'NOT NULL'})")
        
        print("\n📊 Final ChatMessage table schema:")
        cur.execute("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'chat_messages' 
            ORDER BY ordinal_position
        """)
        
        for row in cur.fetchall():
            print(f"  - {row[0]}: {row[1]} ({'NULL' if row[2] == 'YES' else 'NOT NULL'})")
        
        cur.close()
        conn.close()
        
        print("\n🎉 DATABASE FIXED! Your AI chat should now work!")
        return True
        
    except Exception as e:
        print(f"❌ Database fix failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_database()
    if not success:
        exit(1)
