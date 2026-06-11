import os
import re
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()
language_key = os.getenv("LANGUAGE_KEY")
language_endpoint = os.getenv("LANGUAGE_ENDPOINT")

def authenticate_client():
    try:
        credential = AzureKeyCredential(language_key)
        text_analytics_client = TextAnalyticsClient(endpoint=language_endpoint, credential=credential)
        return text_analytics_client
    except Exception as e:
        print(f"Error authenticating client: {e}")
        return None
    

client = authenticate_client()

def analyze_sentiment(client, text):
    documents = [text]

    result = client.analyze_sentiment(documents,
                        show_opinion_mining=True)
    
    doc_result = [doc for doc in result if not doc.is_error]

    for document in doc_result:
        print(f"\n📊 SENTIMENT ANALYSIS RESULTS")
        print(f"Overall Sentiment: {document.sentiment.upper()}")
        print(f"Confidence Scores:")
        print(f"  Positive: {document.confidence_scores.positive:.2f}")
        print(f"  Neutral: {document.confidence_scores.neutral:.2f}")
        print(f"  Negative: {document.confidence_scores.negative:.2f}")

        print(f"\n--- Sentence Breakdown ---")
        for sentence in document.sentences:
            print(f"\nSentence: {re.sub(r'\\s+', ' ', sentence.text.strip())}")
            print(f"Sentiment: {sentence.sentiment}")
            
            for mined_opinion in sentence.mined_opinions:
                target = mined_opinion.target
                print(f"  Target: '{target.text}' → {target.sentiment}")
                for assessment in mined_opinion.assessments:
                    print(f"  Assessment: '{assessment.text}' → {assessment.sentiment}")


if __name__ == "__main__":
    test_text = """
    I recently stayed at the Sunset Palms Hotel in Miami Beach, Florida 
    for five nights in September with my family, and it was one of the 
    most disappointing travel experiences I have ever had. The check-in 
    process was painfully slow, and the front desk staff seemed overwhelmed 
    and uninterested. The Wi-Fi was completely unreliable, housekeeping only 
    visited twice, and the restaurant was overpriced and underwhelming. 
    I contacted the manager Sarah Williams twice but received no meaningful 
    response. I would strongly advise anyone to look elsewhere.
    """
    
    analyze_sentiment(client, test_text)