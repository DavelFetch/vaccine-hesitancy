from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QdrantSetup:
    def __init__(self):
        # Get Qdrant cloud credentials from environment variables
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY environment variables")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Collection configuration
        self.collection_name = "vaccine_guidelines"
        self.vector_size = 1536  # OpenAI ada-002 embedding size
        
    def create_collection(self):
        """Create the collection with proper configuration"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(collection.name == self.collection_name for collection in collections)
            
            if exists:
                print(f"Collection '{self.collection_name}' already exists")
                return
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    on_disk=True  # Store vectors on disk for larger datasets
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                    indexing_threshold=20000
                )
            )
            
            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="title",
                field_schema=models.PayloadSchemaType.TEXT
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="publication_date",
                field_schema=models.PayloadSchemaType.DATETIME
            )
            
            print(f"Successfully created collection '{self.collection_name}' with indexes")
            
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise

if __name__ == "__main__":
    # Create new collection
    setup = QdrantSetup()
    setup.collection_name = "vaccine_guidelines_2"  # Set new collection name
    setup.create_collection()
    print("✅ New Qdrant collection 'vaccine_guidelines_2' created!") 