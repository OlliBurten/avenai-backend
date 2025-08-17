#!/usr/bin/env python3
"""
Test database connection and basic operations
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

def test_database_connection():
    """Test if we can connect to the database"""
    try:
        from database import engine, Base
        from models import Company, User
        
        print("🔍 Testing database connection...")
        
        # Test connection
        with engine.connect() as conn:
            print("✅ Database connection successful!")
        
        # Create tables
        print("🔨 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully!")
        
        # Test basic operations
        from database import SessionLocal
        db = SessionLocal()
        
        try:
            # Count companies
            company_count = db.query(Company).count()
            print(f"📊 Companies in database: {company_count}")
            
            # Count users
            user_count = db.query(User).count()
            print(f"👥 Users in database: {user_count}")
            
            print("✅ Database operations working correctly!")
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Testing Avenai Database System...")
    success = test_database_connection()
    
    if success:
        print("\n🎉 Database system is working correctly!")
        print("\n📝 Next steps:")
        print("1. Set DATABASE_URL in Railway environment variables")
        print("2. Deploy the updated backend")
        print("3. Test registration and login")
    else:
        print("\n❌ Database system has issues that need to be fixed")
