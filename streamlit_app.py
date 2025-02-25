import streamlit as st
import nltk
import gensim
import numpy as np
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

# Load Reuters corpus and preprocess sentences
sentences = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    tokens = preprocess_text(raw_text)
    sentences.append(tokens)

# Train Word2Vec model
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4)

# Compute average embedding for a list of tokens
def compute_average_embedding(tokens):
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

# Compute document embeddings
document_embeddings = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    doc_tokens = preprocess_text(raw_text)
    doc_embedding = compute_average_embedding(doc_tokens)
    document_embeddings.append((fileid, doc_embedding, raw_text))

# Streamlit app
st.title("Document Retrieval System using Word2Vec")
st.write("Enter a query to find the most relevant documents from the Reuters corpus.")

# Input query
query = st.text_input("Enter your query:")

if query:
    # Preprocess query and compute embedding
    query_tokens = preprocess_text(query)
    query_embedding = compute_average_embedding(query_tokens)

    # Compute similarities
    similarities = []
    for fileid, doc_embedding, raw_text in document_embeddings:
        similarity_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        similarities.append((fileid, similarity_score, raw_text))

    # Sort by similarity and get top N results
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_n = 5
    st.write(f"Top {top_n} relevant documents for the query '{query}':")

    for i in range(top_n):
        fileid, similarity_score, raw_text = sorted_similarities[i]
        st.write(f"**Document ID:** {fileid}")
        st.write(f"**Similarity Score:** {similarity_score:.4f}")
        st.write(f"**Document Content:** {raw_text[:200]}...")  # Show first 200 characters
        st.write("---")