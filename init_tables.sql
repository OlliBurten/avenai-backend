-- AVENAI Database Tables Initialization
-- Run this in your Railway PostgreSQL database

-- Create companies table
CREATE TABLE IF NOT EXISTS companies (
    id VARCHAR PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR PRIMARY KEY,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    company_id VARCHAR NOT NULL REFERENCES companies(id),
    is_admin BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    content TEXT,
    file_path VARCHAR(500),
    file_size INTEGER,
    file_type VARCHAR(100),
    company_id VARCHAR NOT NULL REFERENCES companies(id),
    uploaded_by_user_id VARCHAR NOT NULL REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create chat_sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR PRIMARY KEY,
    company_id VARCHAR NOT NULL REFERENCES companies(id),
    user_id VARCHAR NOT NULL REFERENCES users(id),
    session_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create chat_messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL REFERENCES chat_sessions(id),
    message_type VARCHAR(50) NOT NULL, -- 'user', 'ai', 'system'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_company_id ON users(company_id);
CREATE INDEX IF NOT EXISTS idx_documents_company_id ON documents(company_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_company_id ON chat_sessions(company_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);

-- Insert a test company and user for testing
INSERT INTO companies (id, name, description) 
VALUES ('company_test123', 'Test Company', 'Test company for development') 
ON CONFLICT (id) DO NOTHING;

INSERT INTO users (id, email, password_hash, company_id, is_admin) 
VALUES ('user_test123', 'test@example.com', 'test_hash', 'company_test123', true) 
ON CONFLICT (id) DO NOTHING;

-- Verify tables were created
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;
