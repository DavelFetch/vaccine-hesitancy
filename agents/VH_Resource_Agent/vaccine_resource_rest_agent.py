from uagents import Agent, Context, Model
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent
from pydantic import Field
from datetime import datetime, timezone
from uuid import uuid4
from typing import Optional, List, Dict, Any
import logging
import tempfile
import os
import base64
from document_processor import DocumentProcessor

# REST agent config
VACCINE_RESOURCE_AGENT_ADDRESS = "agent1q0tds2u7q4ak8vj2pd9kn25pauuczm9pvqg50jmstuj36tvrf9c57fmj7hy"

class ChatRequest(Model):
    message: str 

class ChatResponse(Model):
    response: str 

class FileUploadRequest(Model):
    filename: str = Field(..., description="Name of the uploaded file")
    file_data: str = Field(..., description="Base64 encoded file data")
    content_type: str = Field(..., description="MIME type of the file")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class FileUploadResponse(Model):
    success: bool
    message: str
    doc_id: Optional[str] = None
    filename: Optional[str] = None
    chunks_count: Optional[int] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    upload_timestamp: Optional[str] = None
    error: Optional[str] = None

class SupportedTypesResponse(Model):
    supported_types: List[str]
    max_file_size_mb: int = 50

class UploadedDocument(Model):
    doc_id: str
    filename: str
    title: str
    source: str
    source_type: str
    file_type: str
    file_size: int
    upload_timestamp: str
    total_chunks: int
    categories: List[str]

class UploadedDocumentsResponse(Model):
    documents: List[UploadedDocument]
    total_count: int

class DeleteDocumentRequest(Model):
    doc_id: str

class DeleteDocumentResponse(Model):
    success: bool
    message: str
    doc_id: str
    error: Optional[str] = None

agent = Agent(
    name="vaccine_resource_rest_agent",
    port=8006,
    seed="vaccine_resource_rest_agent_seed_2024",
    mailbox=True
)

# Initialize document processor
doc_processor = DocumentProcessor()

@agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info("🚀 Vaccine Resource REST Agent started")
    ctx.logger.info("📂 File upload endpoints available:")
    ctx.logger.info("   • POST /upload - Upload and process documents")
    ctx.logger.info("   • GET /supported-types - Get supported file types")
    ctx.logger.info("   • POST /chat - Chat with vaccine resource agent")

@agent.on_rest_post("/chat", ChatRequest, ChatResponse)
async def handle_chat(ctx: Context, req: ChatRequest) -> ChatResponse:
    ctx.logger.info(f"[REST] Received /chat request: {req.message}")
    ctx.logger.info(f"[REST] Using agent address: {VACCINE_RESOURCE_AGENT_ADDRESS}")
    # Construct ChatMessage
    chat_msg = ChatRequest(message=req.message)
    ctx.logger.info(f"[REST] Outgoing ChatMessage: {chat_msg}")
    # Relay to vaccine resource agent and wait for reply
    try:
        reply_obj = await ctx.send_and_receive(
            VACCINE_RESOURCE_AGENT_ADDRESS,
            chat_msg,
            response_type=ChatResponse,  # Specify expected response type
            timeout=60  # seconds
        )
        ctx.logger.info(f"[REST] Received reply: {reply_obj}")
        # Extract text from reply
        # reply_text = None
        # if hasattr(reply_obj, "response"):
        #     reply_text = reply_obj.response
        # Unpack tuple if needed
        if isinstance(reply_obj, tuple):
            reply_msg, _ = reply_obj
        else:
            reply_msg = reply_obj

        reply_text = getattr(reply_msg, "response", None)
        if not reply_text:
            reply_text = "No response from agent."
        return ChatResponse(response=reply_text)
    except Exception as e:
        ctx.logger.error(f"[REST] Error in send_and_receive: {str(e)}")
        return ChatResponse(response=f"Error: {str(e)}")

@agent.on_rest_post("/upload", FileUploadRequest, FileUploadResponse)
async def handle_file_upload(ctx: Context, req: FileUploadRequest) -> FileUploadResponse:
    """Handle file upload and process it for Qdrant storage"""
    ctx.logger.info(f"📁 Received file upload request: {req.filename}")
    
    try:
        # Decode base64 file data
        try:
            file_data = base64.b64decode(req.file_data)
        except Exception as e:
            ctx.logger.error(f"Error decoding base64 file data: {e}")
            return FileUploadResponse(
                success=False,
                message="Invalid file data encoding",
                error=f"Base64 decode error: {str(e)}"
            )
        
        # Validate file
        is_valid, validation_message = doc_processor.validate_file(req.filename, len(file_data))
        if not is_valid:
            ctx.logger.warning(f"File validation failed: {validation_message}")
            return FileUploadResponse(
                success=False,
                message="File validation failed",
                error=validation_message
            )
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, req.filename)
        
        try:
            # Write file data to temporary file
            with open(temp_file_path, 'wb') as f:
                f.write(file_data)
            
            ctx.logger.info(f"📝 Processing file: {req.filename} ({len(file_data)} bytes)")
            
            # Process the file
            result = doc_processor.process_uploaded_file(
                file_path=temp_file_path,
                filename=req.filename,
                metadata=req.metadata
            )
            
            if result['success']:
                ctx.logger.info(f"✅ Successfully processed {req.filename} - Doc ID: {result['doc_id']}")
                return FileUploadResponse(
                    success=True,
                    message=f"File '{req.filename}' successfully processed and stored",
                    doc_id=result['doc_id'],
                    filename=result['filename'],
                    chunks_count=result['chunks_count'],
                    file_size=result['file_size'],
                    file_type=result['file_type'],
                    upload_timestamp=result['upload_timestamp']
                )
            else:
                ctx.logger.error(f"❌ Failed to process {req.filename}: {result['error']}")
                return FileUploadResponse(
                    success=False,
                    message=f"Failed to process file '{req.filename}'",
                    error=result['error']
                )
        
        finally:
            # Clean up temporary directory
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                os.rmdir(temp_dir)
            except Exception as e:
                ctx.logger.warning(f"Failed to clean up temporary files: {e}")
    
    except Exception as e:
        ctx.logger.error(f"❌ Unexpected error processing file upload: {e}")
        return FileUploadResponse(
            success=False,
            message="Unexpected error occurred during file processing",
            error=str(e)
        )

@agent.on_rest_get("/supported-types", SupportedTypesResponse)
async def get_supported_types(ctx: Context) -> SupportedTypesResponse:
    """Get list of supported file types for upload"""
    ctx.logger.info("📋 Returning supported file types")
    return SupportedTypesResponse(
        supported_types=doc_processor.get_supported_file_types(),
        max_file_size_mb=50
    )

@agent.on_rest_get("/uploaded-documents", UploadedDocumentsResponse)
async def get_uploaded_documents(ctx: Context) -> UploadedDocumentsResponse:
    """Get list of uploaded documents from Qdrant"""
    ctx.logger.info("📚 Returning uploaded documents")
    documents = doc_processor.get_uploaded_documents()
    return UploadedDocumentsResponse(
        documents=documents,
        total_count=len(documents)
    )

@agent.on_rest_post("/delete-document", DeleteDocumentRequest, DeleteDocumentResponse)
async def delete_document(ctx: Context, req: DeleteDocumentRequest) -> DeleteDocumentResponse:
    """Delete an uploaded document from Qdrant"""
    ctx.logger.info(f"🗑️ Deleting document: {req.doc_id}")
    result = doc_processor.delete_uploaded_document(req.doc_id)
    
    if result['success']:
        return DeleteDocumentResponse(
            success=True,
            message=result['message'],
            doc_id=req.doc_id
        )
    else:
        return DeleteDocumentResponse(
            success=False,
            message="Failed to delete document",
            doc_id=req.doc_id,
            error=result['error']
        )

if __name__ == "__main__":
    agent.run() 