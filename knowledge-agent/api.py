from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from agent import graph
from pinecone_client import get_index
from openai import OpenAI
from answer_confidence import simple_answer_confidence
import os
from dotenv import load_dotenv
from typing import Optional, List

# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Assistant API",
    description="Enterprise Knowledge Assistant API for multi-platform integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask (1-1000 characters)")
    user_id: Optional[str] = Field(None, max_length=100, description="Optional user identifier")
    platform: Optional[str] = Field(None, max_length=50, description="Optional platform identifier")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Question cannot be empty or whitespace only")
        if len(v.strip()) < 2:
            raise ValueError("Question must be at least 2 characters long")
        return v.strip()
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v is not None and v.strip() == "":
            raise ValueError("User ID cannot be empty if provided")
        return v.strip() if v else None
    
    @validator('platform')
    def validate_platform(cls, v):
        if v is not None:
            allowed_platforms = ['teams', 'slack', 'whatsapp', 'web', 'mobile', 'api', 'test']
            if v.lower() not in allowed_platforms:
                raise ValueError(f"Platform must be one of: {', '.join(allowed_platforms)}")
            return v.lower()
        return None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the best practices for Java?",
                "user_id": "user123",
                "platform": "teams"
            }
        }

class SourceInfo(BaseModel):
    document: str
    page: Optional[int] = None
    relevance_score: float

class QuestionResponse(BaseModel):
    answer: str
    status: str
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Answer quality confidence (0-1)")
    confidence_category: str = Field(..., description="HIGH, GOOD, MODERATE, or LOW")
    is_from_documents: bool = Field(..., description="Whether answer is from indexed documents")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents used")
    user_id: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int

class HealthResponse(BaseModel):
    status: str
    message: str

# Exception handler for validation errors
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors with 400 Bad Request"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "status_code": 400
        }
    )

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Knowledge Assistant API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "ask": "/ask - POST - Ask a question",
            "health": "/health - GET - Health check"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Knowledge Assistant API is running"
    )

@app.post("/ask", response_model=QuestionResponse, responses={
    400: {"model": ErrorResponse, "description": "Bad Request - Invalid input"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"}
})
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the Knowledge Assistant
    
    - **question**: The question to ask (required, 1-1000 characters)
    - **user_id**: Optional user identifier (max 100 characters)
    - **platform**: Optional platform identifier (teams, slack, whatsapp, web, mobile, api, test)
    
    Returns:
    - **answer**: AI-generated answer
    - **confidence_score**: Confidence level (0-1)
    - **sources**: List of source documents with page numbers and relevance scores
    - **status**: Request status
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get query embedding
        response = client.embeddings.create(
            input=request.question,
            model="text-embedding-3-small"
        )
        query_vec = response.data[0].embedding
        
        # Query Pinecone with metadata
        index = get_index()
        results = index.query(
            vector=query_vec,
            top_k=5,
            include_metadata=True
        )
        
        # Check if we have results
        if not results.get("matches") or len(results["matches"]) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "No Information Found",
                    "detail": "I don't have information about this topic in my knowledge base. Please try rephrasing your question or ask about topics covered in the indexed documents.",
                    "status_code": 400
                }
            )
        
        # Extract sources with scores
        sources = []
        seen_docs = set()
        
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            doc_name = metadata.get("source", "Unknown")
            page_num = metadata.get("page")
            score = match.get("score", 0.0)
            
            # Avoid duplicate documents
            doc_key = f"{doc_name}_{page_num}"
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                sources.append(SourceInfo(
                    document=doc_name,
                    page=page_num,
                    relevance_score=round(score, 4)
                ))
        
        # Get context chunks for agent
        context_chunks = [match["metadata"]["text"] for match in results["matches"]]
        context = "\n".join(context_chunks)
        top_score = results["matches"][0].get("score", 0.0)
        
        # Invoke the agent
        result = graph.invoke({
            "question": request.question,
            "context": context,
            "answer": ""
        })
        
        answer = result["answer"]
        
        # Calculate answer confidence (faithfulness to documents)
        answer_confidence = simple_answer_confidence(
            answer=answer,
            retrieved_chunks=context_chunks,
            retrieval_score=top_score
        )
        
        return QuestionResponse(
            answer=answer,
            status="success",
            confidence_score=answer_confidence["confidence_score"],
            confidence_category=answer_confidence["category"],
            is_from_documents=answer_confidence["is_from_documents"],
            sources=sources[:3],
            user_id=request.user_id
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
