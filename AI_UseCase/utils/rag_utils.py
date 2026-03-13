from models.embeddings import load_embedding_model
embedding_model=load_embedding_model()
documents=[]
embeddings=[]
def add_documents(text_chunks):
  global documents,embeddings
  for chunk in text_chunks:
    vector = embedding_model.encode(chunk)
    documents.append(chunk)
    embeddings.append(vector)

def retrieve_document(query):
  import numpy as np
  query_vector=embedding_model.encode(query)
  scores=[]
  for emb in embeddings:
    score=np.dot(query_vector,emb)
    scores.append(score)
  if not scores:
    return None
  best_index=scores.index(max(scores))
  return documents[best_index]