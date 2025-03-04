import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from mistralai import Mistral, UserMessage

# Set up the API key
os.environ["MISTRAL_API_KEY"] = "CYg5ae9wMYxKUJbdjkrcYvC75JsOcr8C"
api_key = os.environ.get('MISTRAL_API_KEY')}")

# Function to fetch and process the webpage
def fetch_and_process_webpage(url):
    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tag = soup.find("div")
    text = tag.text
    return text

# Function to chunk the text
def chunk_text(text, chunk_size=512):
    return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to get embeddings
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

# Function to search for similar chunks using scikit-learn
def search_similar_chunks(query_embedding, embeddings, k=2):
    distances = euclidean_distances([query_embedding], embeddings)[0]
    sorted_indices = np.argsort(distances)
    return sorted_indices[:k]

# Function to generate response
def generate_response(prompt):
    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=prompt)]
    chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return chat_response.choices[0].message.content

# Streamlit app
st.title("UDST Policies Chatbot")

# Fetch and process the webpage
url = "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and"
text = fetch_and_process_webpage(url)
chunks = chunk_text(text)

# Get embeddings
text_embeddings = get_text_embedding(chunks)
embeddings = np.array([text_embedding.embedding for text_embedding in text_embeddings])

# User interface
policy = st.selectbox("Select a policy", ["Sport and Wellness Facilities", "Other Policy"])
query = st.text_input("Enter your query")

if st.button("Submit"):
    question_embeddings = np.array([get_text_embedding([query])[0].embedding])
    similar_indices = search_similar_chunks(question_embeddings[0], embeddings)
    retrieved_chunk = [chunks[i] for i in similar_indices]
    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer:
    """
    response = generate_response(prompt)
    st.text_area("Answer", response, height=200)
