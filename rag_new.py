import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from mistralai import Mistral, UserMessage

# Set up the API key
os.environ["MISTRAL_API_KEY"] = "CYg5ae9wMYxKUJbdjkrcYvC75JsOcr8C"
api_key = os.environ.get('MISTRAL_API_KEY')

# Original base URL
base_url = "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/"

# List of policy categories
policy_categories = [
    "Academic Annual Leave Policy",
    "Academic Appraisal Policy",
    "Academic Appraisal Procedure",
    "Academic Credentials Policy",
    "Academic Freedom Policy",
    "Academic Members’ Retention Policy",
    "Academic Professional Development",
    "Academic Qualifications Policy",
    "Credit Hour Policy",
    "Intellectual Property Policy",
    "Joint Appointment Policy",
    "Program Accreditation Policy",
    "Examination Policy",
    "Student Conduct Policy",
    "Student Conduct Procedure",
    "Academic Schedule Policy",
    "Academic Scheduling Procedure",
    "Student Attendance Policy",
    "Student Attendance Procedure",
    "Student Appeals Policy",
    "Academic Standing Policy",
    "Academic Standing Procedure",
    "Transfer Policy",
    "Admissions Policy",
    "Admissions Procedure",
    "Final Grade Policy",
    "Final Grade Procedure",
    "Registration Policy",
]

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

# User interface
selected_category = st.selectbox("Select a policy category", policy_categories)

# Dynamically construct the URL
category_url = base_url + selected_category.lower().replace(" ", "-")

# Fetch and process the webpage based on the selected category
try:
    text = fetch_and_process_webpage(category_url)
    chunks = chunk_text(text)

    # Get embeddings
    text_embeddings = get_text_embedding(chunks)
    embeddings = np.array([text_embedding.embedding for text_embedding in text_embeddings])

    # User query
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
except Exception as e:
    st.error(f"Failed to fetch or process the webpage: {e}")
