import streamlit as st
import numpy as np
import os
from mistralai import Mistral, UserMessage

os.environ["MISTRAL_API_KEY"] = "CYg5ae9wMYxKUJbdjkrcYvC75JsOcr8C"
print(f"MISTRAL_API_KEY: {os.environ.get('MISTRAL_API_KEY')}")
api_key = os.getenv("MISTRAL_API_KEY")

def mistral(user_message, model="mistral-small-latest", is_json=False):
    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)
    messages = [
        UserMessage(content=user_message),
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
    )
    return chat_response.choices[0].message.content

st.title("Information Retrieval Using Mistral AI")

query = st.text_input("Ask you questions away :)")

respose = mistral(query)

st.write("### Here are the results to your question:")
st.write(respose)
