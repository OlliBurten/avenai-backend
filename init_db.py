from database import engine, Base
from models import Company, User, Document, ChatSession, ChatMessage
from auth_utils import get_password_hash
import os

def init_database():
    """Initialize the database with tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def create_initial_data():
    """Create initial data for testing"""
    from database import SessionLocal
    
    db = SessionLocal()
    try:
        # Check if we already have data
        existing_company = db.query(Company).first()
        if existing_company:
            print("Database already has data, skipping initialization")
            return
        
        print("Creating initial test data...")
        
        # Create a test company
        test_company = Company(
            name="Avenai Demo Company",
            description="Demo company for testing Avenai platform"
        )
        db.add(test_company)
        db.flush()  # Get the ID
        
        # Create a test user
        test_user = User(
            email="admin@avenai.com",
            password_hash=get_password_hash("admin123"),
            company_id=test_company.id,
            is_admin=True
        )
        db.add(test_user)
        
        # Commit the changes
        db.commit()
        print("Initial test data created successfully!")
        print(f"Test company: {test_company.name}")
        print(f"Test user: {test_user.email} / admin123")
        
    except Exception as e:
        print(f"Error creating initial data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
    create_initial_data()
