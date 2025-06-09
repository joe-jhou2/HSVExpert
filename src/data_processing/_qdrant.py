from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from qdrant_client.models import PointStruct
import numpy as np
import uuid

def create_collection_if_not_exists(client, collection_name, vector_size, distance="Cosine"):
    existing_collections = client.get_collections().collections
    if collection_name not in [c.name for c in existing_collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )

def sanitize_payload(obj):
    if isinstance(obj, dict):
        return {k: sanitize_payload(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_payload(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def store_chunks_in_qdrant(chunks, 
                           embeddings, 
                           embed_model_name="unknown", 
                           collection_name="papers", 
                           source = "pre-processed",
                           host="localhost", 
                           port=6333):
    client = QdrantClient(host=host, port=port)
    
    if not embeddings:  # Check if empty
        print("No embeddings provided. Skipping storing in Qdrant.")
        return

    # Assuming all embeddings have same size
    vector_size = len(embeddings[0])
    
    # Create collection if it doesn't exist
    create_collection_if_not_exists(client, collection_name, vector_size)
    
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        
        payload = sanitize_payload(chunk.copy())
        payload["original_id"] = f"{chunk['paper_id']}_{chunk.get('chunk_id', idx)}"
        payload["embed_model_name"] = embed_model_name
        payload["source"] = source
        
        point_id = f"{chunk['paper_id']}_{chunk.get('chunk_id', idx)}" 
        unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))

        point = PointStruct(
            id=unique_id,
            vector=vector_list,
            payload=payload
        )
        points.append(point)
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )

