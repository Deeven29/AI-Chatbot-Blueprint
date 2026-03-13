from sentence_transformers import SentenceTransformer

def load_embedding_model():
  try:
    model=SentenceTransformer("all-MiniLM-L6-v2")
    return model
  except Exception as e:
    raise RuntimeError(f"Failed to  load embedding model: {str(e)}")
  