
import streamlit as st
import os
import sqlite3
import datetime
import pandas as pd
import requests
from langdetect import detect

# LangChain components
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# External search using duckduckgo_search
from duckduckgo_search import DDGS

# =============================================================================
# MODULE 1: Tatoeba Data Retrieval (Dynamic)
# =============================================================================
def search_tatoeba(query: str, from_lang: str = "eng", to_lang: str = "spa") -> str:
    """
    Query Tatoeba's public REST API for sentence pairs.
    For example, to get English sentences and their Spanish translations for "hello".
    Returns a summary of the top result (first 200 characters).
    """
    url = "https://tatoeba.org/en/api_v0/search"
    params = {
        "query": query,
        "from": from_lang,
        "to": to_lang,
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            # Assume data has a key 'results' with a list of sentence pairs.
            results = data.get("results", [])
            if results:
                # For simplicity, concatenate the first result's source and target sentences.
                first_result = results[0]
                source = first_result.get("source", "")
                target = first_result.get("target", "")
                summary = f"Example: {source} --> {target}"
                return summary[:200]
            else:
                return "No Tatoeba results found."
        else:
            return f"Tatoeba API error: {response.status_code}"
    except Exception as e:
        return f"Tatoeba search error: {e}"

# =============================================================================
# MODULE 2: External Search Integration using DuckDuckGo
# =============================================================================
def external_search(query: str) -> str:
    """
    Uses DDGS from duckduckgo_search to fetch live search results.
    Returns the first 200 characters of the result's body.
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=1)
            if results:
                summary = results[0].get("body", "")
                return summary[:200]
            else:
                return "No external search information found."
    except Exception as e:
        return f"External search error: {e}"

# =============================================================================
# MODULE 3: Build the Vector Store with a Multilingual Lesson Corpus
# =============================================================================
# Instead of using a static CSV, we now assume you have a dataset of curated lessons.
# For demonstration, we create fallback dummy data.
from langchain.docstore.document import Document

def load_lessons_from_api() -> list:
    """
    This function simulates dynamic lesson data retrieval.
    In a real-world scenario, you might query an API or database.
    """
    # For demonstration, return a list of Document objects.
    dummy_lessons = [
        {"language": "german", "lesson_text": "German Lesson: 'Hallo' means 'hello'."},
        {"language": "french", "lesson_text": "French Lesson: 'Bonjour' means 'hello'."},
        {"language": "spanish", "lesson_text": "Spanish Lesson: 'Hola' is a basic greeting."},
        {"language": "italian", "lesson_text": "Italian Lesson: 'Ciao' is used for greeting."},
        {"language": "chinese", "lesson_text": "Chinese Lesson: '你好' (nǐ hǎo) means 'hello'."},
        {"language": "japanese", "lesson_text": "Japanese Lesson: 'こんにちは' (konnichiwa) means 'hello'."},
        {"language": "russian", "lesson_text": "Russian Lesson: 'Привет' (privet) is a common greeting."},
    ]
    documents = []
    for item in dummy_lessons:
        documents.append(Document(page_content=item["lesson_text"], metadata={"language": item["language"]}))
    return documents

documents = load_lessons_from_api()

from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs_split = []
for doc in documents:
    docs_split.extend(splitter.split_documents([doc]))

if not docs_split:
    st.error("No documents available for embeddings.")
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs_split, embedding=embeddings)

# =============================================================================
# MODULE 4: Retrieval Module (RAG)
# =============================================================================
generator = pipeline("text-generation", model="gpt2-medium", max_new_tokens=100)
llm_hf = HuggingFacePipeline(pipeline=generator)
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm_hf,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 1})
)

# =============================================================================
# MODULE 5: Conversation Engine & Adaptive Prompting
# =============================================================================
def clean_context(context: str) -> str:
    sentences = context.split(". ")
    for sentence in sentences:
        if "Lesson:" in sentence:
            return sentence.strip() + "."
    return context.strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a friendly, adaptive language-learning assistant for absolute beginners. "
     "Explain concepts in simple terms using basic vocabulary and provide clear explanations. "
     "Incorporate lesson context and additional information (from external searches or Tatoeba) without exposing raw data."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{human_input}")
])
llm_chat = llm_hf

def get_response(user_input: str, history: list) -> str:
    # Retrieve lesson context.
    raw_context = retrieval_qa.invoke({"query": user_input})
    retrieval_context = clean_context(raw_context)
    # Fetch dynamic examples from Tatoeba; adjust language parameters as needed.
    tatoeba_info = search_tatoeba(user_input, from_lang="eng", to_lang="fra")  # Example: English to French
    # Fetch additional info via external search.
    ext_info = external_search(user_input)
    combined_context = f"{retrieval_context}\nTatoeba: {tatoeba_info}\nExtra Info: {ext_info}"
    combined_input = f"Context: {combined_context}\nUser: {user_input}"
    formatted_prompt = prompt.format(human_input=combined_input, history=history)
    response = llm_chat(formatted_prompt)
    return response.strip()

# =============================================================================
# MODULE 6: Error Detection & Logging
# =============================================================================
conn = sqlite3.connect("chatbot_errors.db")
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS mistakes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_input TEXT,
        error TEXT,
        suggestion TEXT
    )
''')
conn.commit()

def detect_errors(text: str) -> tuple:
    if "teh" in text:
        return "Spelling error: 'teh' should be 'the'", "Please check your spelling."
    if text.isupper():
        return "Text is all uppercase", "Use proper capitalization."
    return None, None

def log_error(user_input: str, error: str, suggestion: str):
    timestamp = datetime.datetime.now().isoformat()
    c.execute("INSERT INTO mistakes (timestamp, user_input, error, suggestion) VALUES (?, ?, ?, ?)",
              (timestamp, user_input, error, suggestion))
    conn.commit()

def get_logged_errors() -> list:
    cursor = conn.execute("SELECT timestamp, user_input, error, suggestion FROM mistakes")
    return cursor.fetchall()

def generate_review() -> str:
    records = get_logged_errors()
    review_text = "Review Summary:\n"
    for rec in records:
        review_text += f"{rec[0]} - You wrote: '{rec[1]}'. Issue: {rec[2]}. Suggestion: {rec[3]}\n"
    return review_text

# =============================================================================
# MODULE 7: Language Detection & Adaptive Scene Selection
# =============================================================================
def detect_language_of_input(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def user_initialization() -> tuple:
    st.sidebar.header("User Preferences")
    target_language = st.sidebar.selectbox("Which language do you want to learn?", 
                                           options=["German", "French", "Spanish", "Italian", "Chinese", "Japanese", "Russian"])
    known_language = st.sidebar.selectbox("Which language do you know?", options=["English", "Other"])
    proficiency = st.sidebar.selectbox("Your proficiency level:", options=["Beginner", "Intermediate", "Advanced"])
    return target_language, known_language, proficiency

def select_scene(target_language: str, proficiency: str) -> str:
    if target_language.lower() == "german" and proficiency.lower() == "beginner":
        return "Let's have a simple conversation in German with basic greetings."
    elif target_language.lower() == "french" and proficiency.lower() == "beginner":
        return "Let's start with simple greetings in French."
    elif target_language.lower() == "spanish" and proficiency.lower() == "beginner":
        return "Let's simulate a conversation in a Spanish restaurant where you'll order food from scratch."
    else:
        return f"Let's have a conversation in {target_language} on a simple, general topic."

# =============================================================================
# MODULE 8: Streamlit Web Interface
# =============================================================================
st.title("Real-World Adaptive Language Learning Chatbot")

target_language, known_language, proficiency = user_initialization()
st.write(f"Great! You want to learn **{target_language}** and you know **{known_language}**. Proficiency: **{proficiency}**.")

scene_info = select_scene(target_language, proficiency)
st.write("**Scene Information:**", scene_info)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [HumanMessage(content=f"Scene: {scene_info}")]

user_input = st.text_input("You:")
if st.button("Send"):
    if user_input:
        detected_lang = detect_language_of_input(user_input)
        st.write(f"Detected input language: {detected_lang}")
        error, suggestion = detect_errors(user_input)
        if error:
            st.error(f"[Error Detected] {error}. Suggestion: {suggestion}")
            log_error(user_input, error, suggestion)
        response = get_response(user_input, st.session_state.chat_history)
        st.write("**Chatbot:**", response)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.user_input = ""

if st.button("Show Review Summary"):
    review = generate_review()
    st.text_area("Review Summary", review, height=200)

st.markdown("""
**Future Enhancements:**
- **Adaptive Difficulty:** Adjust conversation complexity based on learner performance.
- **Gamification:** Award points, badges, and levels.
- **Spaced Repetition:** Recommend review sessions for frequent mistakes.
- **Live Internet Search:** Integrate a dedicated search API for up-to-date explanations.
- **Personalized Feedback:** Track progress and suggest tailored learning paths.
""")
