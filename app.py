import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from transformers import pipeline
import streamlit as st

# --- Config ---
MODEL_NAME = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")
@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model=MODEL_NAME)

# --- Helper functions ---
def extract_text_from_url(url, timeout=20):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Gather paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
        article_text = " ".join(paragraphs)

        # Gather images
        image_urls = []
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if src and not src.startswith("data:"):
                image_urls.append(urljoin(url, src))

        return article_text, list(dict.fromkeys(image_urls))
    except Exception:
        return "", []


def chunk_text(text, max_chars=1000):
    import re
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    chunks, current, length = [], [], 0
    for s in sentences:
        if length + len(s) <= max_chars or not current:
            current.append(s)
            length += len(s)
        else:
            chunks.append(" ".join(current))
            current, length = [s], len(s)
    if current: chunks.append(" ".join(current))
    return chunks

def summarize_long_text(text):
    summarizer = get_summarizer()
    if len(text) <= 1000:
        return summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]

    chunks = chunk_text(text, max_chars=1000)
    summaries = []
    for c in chunks:
        try:
            summaries.append(
                summarizer(c, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
            )
        except Exception:
            continue
    if not summaries:
        return text[:400] + "..."
    combined = " ".join(summaries)
    return summarizer(combined, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]

# --- Streamlit UI ---
st.set_page_config(page_title="Fast Fresh Facts", layout="wide")
st.title("⚡ Fast Fresh Facts")
st.write("Paste a news article URL and get a short summary.")

url = st.text_input("Enter article URL:")
if st.button("Summarize") and url:
    with st.spinner("Fetching and summarizing..."):
        text, images = extract_text_from_url(url)
        if not text or len(text) < 100:
            st.error("Could not extract enough text to summarize.")
        else:
            summary = summarize_long_text(text)
            st.subheader("Summary")
            st.write(summary)

            if images:
                st.subheader("Images")
                st.image(images, width=250)
