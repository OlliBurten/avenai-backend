#!/usr/bin/env python3
"""
Startup script for Avenai Backend
Runs database migration and then starts the main application
"""

import os
import sys
import subprocess
import time

def run_migration():
    """Run database migration"""
    print("🔧 Running database migration...")
    try:
        from fix_database_schema import migrate_database
        success = migrate_database()
        if success:
            print("✅ Migration completed successfully!")
            return True
        else:
            print("❌ Migration failed!")
            return False
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False

def start_app():
    """Start the main FastAPI application"""
    print("🚀 Starting Avenai Backend...")
    try:
        # Import and run the main app
        from avenai_final import app
        import uvicorn
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        sys.exit(1)

def main():
    """Main startup sequence"""
    print("🚀 Avenai Backend Starting Up...")
    print("=" * 40)
    
    # Wait a bit for database to be ready
    print("⏳ Waiting for database to be ready...")
    time.sleep(5)
    
    # Run migration
    if not run_migration():
        print("⚠️  Migration failed, but continuing...")
    
    # Start the app
    start_app()

if __name__ == "__main__":
    main()
