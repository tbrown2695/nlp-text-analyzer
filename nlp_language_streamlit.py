import os
import re
import streamlit as st
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from collections import Counter

# Load environment variables
load_dotenv()
language_key = os.getenv("LANGUAGE_KEY")
language_endpoint = os.getenv("LANGUAGE_ENDPOINT")

# Page configuration
st.set_page_config(
    page_title="NLP Text Analyzer",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 NLP Text Analyzer")
st.markdown("Analyze sentiment and opinions in text using Azure AI Language")
st.divider()

# Authenticate client
def authenticate_client():
    try:
        credential = AzureKeyCredential(language_key)
        client = TextAnalyticsClient(
            endpoint=language_endpoint,
            credential=credential
        )
        return client
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return None

client = authenticate_client()

# Text input
st.subheader("📝 Enter Text to Analyze")
user_text = st.text_area(
    label="Paste your text here",
    height=200,
    placeholder="Enter a review, article, or any text you want to analyze..."
)

analyze_button = st.button("🔍 Analyze Sentiment", type="primary")

def analyze_sentiment(client, text):
    documents = [text]
    
    result = client.analyze_sentiment(
        documents,
        show_opinion_mining=True
    )
    
    doc_result = [doc for doc in result if not doc.is_error]
    
    for document in doc_result:
        # Overall sentiment
        sentiment = document.sentiment.upper()
        
        if sentiment == "POSITIVE":
            st.success(f"📊 Overall Sentiment: {sentiment}")
        elif sentiment == "NEGATIVE":
            st.error(f"📊 Overall Sentiment: {sentiment}")
        else:
            st.warning(f"📊 Overall Sentiment: {sentiment}")
        
        # Confidence scores
        st.subheader("Confidence Scores")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Positive", 
                f"{document.confidence_scores.positive:.0%}")
        with col2:
            st.metric("Neutral", 
                f"{document.confidence_scores.neutral:.0%}")
        with col3:
            st.metric("Negative", 
                f"{document.confidence_scores.negative:.0%}")
        
        st.divider()
        
        # Sentence breakdown
        st.subheader("📝 Sentence Breakdown")
        for sentence in document.sentences:
            with st.expander(
                f"{sentence.sentiment.upper()}: {sentence.text.strip()[:80]}..."
            ):
                st.write(f"**Sentiment:** {sentence.sentiment}")
                
                if sentence.mined_opinions:
                    st.write("**Opinions found:**")
                    for mined_opinion in sentence.mined_opinions:
                        target = mined_opinion.target
                        for assessment in mined_opinion.assessments:
                            st.write(
                                f"  • **{target.text}** → "
                                f"_{assessment.text}_ "
                                f"({assessment.sentiment})"
                            )


def extract_key_phrases(client, text):
    documents = [text]
    result = client.extract_key_phrases(documents)
    for doc in result:
        if not doc.is_error:
            st.subheader("🔑 Key Phrases")
            phrases = doc.key_phrases
            cols = st.columns(3)
            for i, phrase in enumerate(phrases):
                cols[i % 3].markdown(f"• {phrase}")

def recognize_entities(client, text):
    documents = [text]
    result = client.recognize_entities(documents)
    for doc in result:
        if not doc.is_error:
            st.subheader("🏷️ Named Entities")
            for entity in doc.entities:
                st.markdown(
                    f"**{entity.text}** — "
                    f"_{entity.category}_ "
                    f"({entity.confidence_score:.0%} confidence)"
                )

def summarize_text(client, text):
    documents = [{"id": "1", "language": "en", "text": text}]
    poller = client.begin_extract_summary(documents)
    result = poller.result()
    for doc in result:
        if not doc.is_error:
            st.subheader("📋 Summary")
            summary = " ".join([s.text for s in doc.sentences])
            st.info(summary)

def calculate_term_frequency(text):
    st.subheader("📊 Term Frequency")
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stop_words = {
        'the', 'and', 'was', 'for', 'that', 'this', 
        'with', 'had', 'were', 'have', 'but', 'not', 
        'from', 'they', 'our', 'which', 'who', 'would', 
        'could', 'their', 'been', 'has', 'its'
    }
    words = [w for w in words if w not in stop_words]
    counter = Counter(words)
    top_words = counter.most_common(10)
    for word, count in top_words:
        st.markdown(f"**{word}**: {count}")



# Run analysis when button is clicked
if analyze_button:
    if not user_text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    elif client is None:
        st.error("❌ Could not connect to Azure. Check your credentials.")
    else:
        with st.spinner("🔍 Analyzing text..."):
            
            st.divider()
            
            # Sentiment Analysis
            analyze_sentiment(client, user_text)
            
            st.divider()
            
            # Key Phrases
            extract_key_phrases(client, user_text)
            
            st.divider()
            
            # Named Entities
            recognize_entities(client, user_text)
            
            st.divider()
            
            # Summarization
            summarize_text(client, user_text)
            
            st.divider()
            
            # Term Frequency
            calculate_term_frequency(user_text)
            