import streamlit as st
import requests
from bs4 import BeautifulSoup
import cohere

# Load environment variables
cohere_api_key = "pTJLz8Znb6T5faYpxEfwIkyY5gdLJ8wSzaxMW1Rm"

# Initialize CoHere client
co = cohere.Client(cohere_api_key)

# Function to fetch text content from URL
def fetch_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text()
        return text_content
    else:
        st.error(f"Failed to load content from {url}. Status code: {response.status_code}")
        return None

# Function to search for answer to question within text content
def search_answer(url_list, question):
    best_answer = None
    best_score = float('-inf')
    
    for url in url_list:
        text_content = fetch_text_from_url(url)
        if text_content:
            # Get embeddings for text content
            response = co.embed(texts=[text_content]).embeddings
            # Prepare prompt for question answering
            prompt = f"""
            Excerpt from the article:
            {text_content}
            Question: {question}

            Extract the answer to the question from the text provided.
            If the text doesn't contain the answer, reply that the answer is not available.
            """
            # Generate answer using CoHere
            prediction = co.generate(
                prompt=prompt,
                max_tokens=70,
                model="command-nightly",
                temperature=0.5,
                num_generations=1
            )
            answer_text = prediction.generations[0].text
            
            # Score the answer
            score = compute_answer_score(answer_text, question)
            if score > best_score:
                best_score = score
                best_answer = answer_text
    
    return best_answer

def compute_answer_score(answer, question):
    # Your scoring mechanism goes here
    # For example, you could use string similarity or NLP metrics
    # Here, we'll just return a constant score for demonstration purposes
    return 1.0

# Streamlit UI
st.title("URL Question Answering System")
url1 = st.text_input("Enter URL 1:")
url2 = st.text_input("Enter URL 2:")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if not url1 or not url2:
        st.error("Please enter both URLs.")
    elif not question:
        st.error("Please enter a question.")
    else:
        answer = search_answer([url1, url2], question)
        if answer:
            st.success("Answer:")
            st.write(answer)
        else:
            st.error("Failed to fetch content from the provided URLs.")
