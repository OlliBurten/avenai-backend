#!/usr/bin/env python3
"""
Simple Avenai - Clean, Working Version
Focus: PDF Upload + Text Extraction + AI Chat
"""

import os
import io
import uuid
from datetime import datetime
from typing import Optional, List, Dict
import json

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
    # Initialize client only if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = openai.OpenAI(api_key=api_key)
    else:
        OPENAI_AVAILABLE = False
        client = None
        print("⚠️  No OPENAI_API_KEY found in environment")
except Exception as e:
    print(f"⚠️  OpenAI initialization error: {e}")
    OPENAI_AVAILABLE = False
    client = None

# ============================================================================
# SIMPLE STORAGE (In-Memory for Now)
# ============================================================================

# Simple document storage
DOCUMENTS = {}  # doc_id -> document_data
CHAT_SESSIONS = {}  # session_id -> chat_data

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_pdf_text(pdf_content: bytes) -> str:
    """Extract text from PDF content"""
    if not PDF_AVAILABLE:
        return "[PDF Document] - PyPDF2 not available"
    
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += f"--- Page {page_num + 1} ---\n"
            text_content += page.extract_text() + "\n\n"
        
        print(f"✅ PDF text extracted: {len(text_content)} characters")
        return text_content
        
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        return f"[PDF Document] - Text extraction failed: {str(e)}"

def get_ai_response(user_message: str, document_content: str = None) -> str:
    """Get AI response using OpenAI"""
    if not OPENAI_AVAILABLE or not client:
        return "OpenAI not available. Please check your API key."
    
    try:
        # Build system prompt with document context
        if document_content:
            system_prompt = f"""You are Avenai, an AI-powered API integration support specialist.

You have access to the following document content:
{document_content}

Answer questions based on this document content. Be specific and helpful."""
        else:
            system_prompt = "You are Avenai, an AI-powered API integration support specialist. Help with API questions."
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return f"Error getting AI response: {str(e)}"

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Simple Avenai", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "message": "Simple Avenai is running!",
        "pdf_available": PDF_AVAILABLE,
        "openai_available": OPENAI_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    company_id: str = Form("default_company")
):
    """Upload and process a document"""
    try:
        # Generate document ID
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # Read file content
        file_content = await file.read()
        
        # Extract text based on file type
        if file.filename.lower().endswith('.pdf'):
            content_text = extract_pdf_text(file_content)
        elif file.filename.lower().endswith(('.txt', '.md')):
            content_text = file_content.decode('utf-8')
        else:
            content_text = f"[Document: {file.filename}] - Unsupported file type"
        
        # Store document
        document_data = {
            "id": doc_id,
            "filename": file.filename,
            "file_size": len(file_content),
            "content": content_text,
            "content_length": len(content_text),
            "company_id": company_id,
            "uploaded_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        DOCUMENTS[doc_id] = document_data
        
        print(f"✅ Document uploaded: {doc_id} - {len(content_text)} characters")
        
        return {
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "content_length": len(content_text),
            "message": "Document uploaded and processed successfully"
        }
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/v1/documents")
async def list_documents():
    """List all documents"""
    docs = []
    for doc_id, doc_data in DOCUMENTS.items():
        docs.append({
            "id": doc_id,
            "filename": doc_data["filename"],
            "content_length": doc_data["content_length"],
            "status": doc_data["status"],
            "uploaded_at": doc_data["uploaded_at"]
        })
    
    return {"documents": docs, "total": len(docs)}

@app.post("/api/v1/ai-chat/chat")
async def chat_with_ai(
    message: str = Form(...),
    document_id: Optional[str] = Form(None)
):
    """Chat with AI using document context"""
    try:
        # Get document content if specified
        document_content = None
        if document_id and document_id in DOCUMENTS:
            document_content = DOCUMENTS[document_id]["content"]
            print(f"📚 Using document context: {document_id} - {len(document_content)} characters")
        else:
            print("📚 No document context provided")
        
        # Get AI response
        ai_response = get_ai_response(message, document_content)
        
        return {
            "success": True,
            "response": ai_response,
            "document_used": document_id if document_id else None
        }
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    if document_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")
    
    del DOCUMENTS[document_id]
    return {"success": True, "message": "Document deleted successfully"}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Simple Avenai...")
    print(f"📄 PDF Processing: {'✅ Available' if PDF_AVAILABLE else '❌ Not Available'}")
    print(f"🤖 OpenAI: {'✅ Available' if OPENAI_AVAILABLE else '❌ Not Available'}")
    
    uvicorn.run(
        "simple_avenai:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
# FORCE RAILWAY TO USE NEW CONFIG - Tue Aug 19 10:18:42 CEST 2025
