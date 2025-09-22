import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import time

# --- 1. Train a Simple Model and Vectorizer within the script ---
# The model and vectorizer are created and cached to improve performance.
@st.cache_resource
def get_model_pipeline():
    # A larger, more balanced dummy dataset for demonstration
    # Approximately 1000 entries
    positive_base = [
        "This product is amazing and works perfectly!",
        "I love the service, it's outstanding.",
        "Everything was great, I would recommend it.",
        "I am feeling very happy and energetic.",
        "The customer support was fantastic!",
        "This is a truly magnificent product.",
        "The weather is beautiful today.",
        "I'm so glad I found this place.",
        "What a wonderful experience!",
        "This is the best purchase I have ever made."
    ]

    negative_base = [
        "The performance is terrible and very slow.",
        "I am so disappointed with this experience.",
        "I hate this, it's the worst thing I've ever bought.",
        "What a terrible movie, it was a waste of time.",
        "This is a bad day.",
        "I hate the service, it was horrible.",
        "This is a terrible mistake.",
        "It was a bad decision to go there.",
        "I'm so frustrated with this app.",
        "The worst experience of my life."
    ]

    # Combine and expand the dataset to reach ~1000 entries
    data_list = []
    
    # Positive Sentences
    for i in range(500):
        text_template = positive_base[i % len(positive_base)]
        data_list.append({'text': text_template, 'sentiment': 1})
    
    # Negative Sentences
    for i in range(500):
        text_template = negative_base[i % len(negative_base)]
        data_list.append({'text': text_template, 'sentiment': 0})

    # Add some mixed sentiment sentences to improve robustness
    data_list.extend([
        {'text': "The food was good, but the service was slow.", 'sentiment': 0},
        {'text': "The product is not bad, it's actually pretty good.", 'sentiment': 1},
        {'text': "It was okay, nothing special.", 'sentiment': 1},
        {'text': "The food was great but the long wait was frustrating.", 'sentiment': 0}
    ])
    
    df = pd.DataFrame(data_list)

    # Create a pipeline that combines the vectorizer and the classifier
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Fit the pipeline to the data
    pipeline.fit(df['text'], df['sentiment'])

    return pipeline

# Get the trained pipeline (vectorizer and model combined)
model_pipeline = get_model_pipeline()

# --- 2. Set up the Streamlit page with a beautiful design ---
st.set_page_config(
    page_title="Interactive Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a more polished look
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.st-emotion-cache-1xw2khv {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-right: 1.5rem;
    padding-left: 1.5rem;
}
.big-font {
    font-size: 24px !important;
    font-weight: bold;
}
.prediction-box {
    border: 2px solid #4CAF50;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-top: 20px;
    background-color: #e8f5e9;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}
.negative-box {
    border: 2px solid #F44336;
    background-color: #ffebee;
}
.neutral-box {
    border: 2px solid #FFC107;
    background-color: #fff8e1;
}
.st-emotion-cache-1xw2khv > .st-emotion-cache-163m43n {
    display: flex;
    justify-content: center;
}
.positive-word {
    color: #2e7d32;
    font-weight: bold;
}
.negative-word {
    color: #d32f2f;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Use a container for the main content
with st.container():
    st.title("💡 Sentiment Analysis Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Enter Text for Analysis")
        user_text = st.text_area(
            "Type your text here",
            "This is an example text. I am very happy with this product.",
            height=200,
            key="user_text_input"
        )
        st.markdown("") # Adds some vertical space
        predict_button = st.button("Analyze Sentiment")
    
    with col2:
        st.header("About the Model")
        with st.expander("Click to learn more about this demo"):
            st.markdown(
                """
                This dashboard uses a **Random Forest Classifier** trained on a small, dummy dataset to predict sentiment.
                
                * **Positive Sentiment (1):** Text with a positive tone.
                * **Negative Sentiment (0):** Text with a negative tone.
                
                The model is very basic and is intended for demonstration purposes only.
                """
            )

st.markdown("---")

# --- 3. Main area for prediction with interactive elements ---
if predict_button:
    if not user_text:
        st.error("Please enter some text to analyze.")
    else:
        # Create a loading spinner while the prediction is running
        with st.spinner("Analyzing text..."):
            time.sleep(1) # Simulate a short delay
            try:
                # The pipeline automatically transforms the text and makes a prediction
                prediction = model_pipeline.predict([user_text])
                prediction_proba = model_pipeline.predict_proba([user_text])

                # Get the sentiment and confidence
                sentiment_map = {0: "Negative 😔", 1: "Positive 😊"}
                predicted_sentiment = sentiment_map.get(prediction[0], "Neutral 😐")
                confidence = prediction_proba[0][prediction[0]]

                # Display the prediction with a more visually appealing layout
                st.subheader("Prediction Result")
                
                # Use a custom div for styling
                box_style = "prediction-box"
                if predicted_sentiment.startswith("Negative"):
                    box_style += " negative-box"
                elif predicted_sentiment.startswith("Neutral"):
                    box_style += " neutral-box"

                st.markdown(f'<div class="{box_style}">', unsafe_allow_html=True)
                st.markdown(f'<p class="big-font"><b>{predicted_sentiment}</b></p>', unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)

                # Add a message and visual effect based on sentiment
                if predicted_sentiment.startswith("Positive"):
                    st.balloons()
                    st.success("The model is confident this text is positive!")
                elif predicted_sentiment.startswith("Negative"):
                    st.warning("The model predicts this text is negative.")
                else:
                    st.info("The sentiment is neutral or ambiguous.")
                
                # --- New Feature: Keyword Highlighting ---
                st.subheader("Keyword Analysis")
                
                # Simple keyword lists for highlighting
                positive_keywords = ["happy", "love", "great", "amazing", "perfectly", "outstanding", "good", "recommend", "fantastic", "energetic", "magnificent", "beautiful"]
                negative_keywords = ["terrible", "slow", "disappointed", "hate", "worst", "bad", "horrible", "frustrated", "sad", "mistake"]

                words = user_text.split()
                highlighted_text = []

                for word in words:
                    clean_word = word.lower().strip(".,!?;:\"'")
                    if clean_word in positive_keywords:
                        highlighted_text.append(f'<span class="positive-word">{word}😊</span>')
                    elif clean_word in negative_keywords:
                        highlighted_text.append(f'<span class="negative-word">{word}😔</span>')
                    else:
                        highlighted_text.append(word)

                st.markdown(" ".join(highlighted_text), unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
