#!/usr/bin/env python3
"""
Database Constraints Fix Script
Fixes NOT NULL constraints on old columns that are no longer used
"""

import psycopg2

# Database connection
DATABASE_URL = "postgresql://postgres:TLqDHVnPDUFqRZgaCiXbPPmtyskKvdrH@ballast.proxy.rlwy.net:16651/railway"

def fix_constraints():
    """Fix database constraints"""
    print("🔧 Fixing Database Constraints...")
    print("=" * 40)
    
    try:
        # Connect to database
        print("🔌 Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        print("✅ Connected to database!")
        
        # Fix ChatSession table constraints
        print("\n📝 Fixing ChatSession table constraints...")
        
        # Make session_token nullable (since we don't use it anymore)
        try:
            cur.execute("""
                ALTER TABLE chat_sessions 
                ALTER COLUMN session_token DROP NOT NULL
            """)
            print("✅ Made session_token nullable")
        except Exception as e:
            print(f"⚠️  Error with session_token: {e}")
        
        # Make user_id nullable (since we use created_by now)
        try:
            cur.execute("""
                ALTER TABLE chat_sessions 
                ALTER COLUMN user_id DROP NOT NULL
            """)
            print("✅ Made user_id nullable")
        except Exception as e:
            print(f"⚠️  Error with user_id: {e}")
        
        # Make last_activity nullable (since we use updated_at now)
        try:
            cur.execute("""
                ALTER TABLE chat_sessions 
                ALTER COLUMN last_activity DROP NOT NULL
            """)
            print("✅ Made last_activity nullable")
        except Exception as e:
            print(f"⚠️  Error with last_activity: {e}")
        
        # Fix ChatMessage table constraints
        print("\n📝 Fixing ChatMessage table constraints...")
        
        # Make message_type nullable (since we use role now)
        try:
            cur.execute("""
                ALTER TABLE chat_messages 
                ALTER COLUMN message_type DROP NOT NULL
            """)
            print("✅ Made message_type nullable")
        except Exception as e:
            print(f"⚠️  Error with message_type: {e}")
        
        # Make created_at nullable (since we use timestamp now)
        try:
            cur.execute("""
                ALTER TABLE chat_messages 
                ALTER COLUMN created_at DROP NOT NULL
            """)
            print("✅ Made created_at nullable")
        except Exception as e:
            print(f"⚠️  Error with created_at: {e}")
        
        # Commit changes
        conn.commit()
        print("\n✅ All constraint changes committed!")
        
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
        
        print("\n🎉 Database constraints fixed! Your AI chat should now work!")
        return True
        
    except Exception as e:
        print(f"❌ Constraint fix failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_constraints()
    if not success:
        exit(1)
