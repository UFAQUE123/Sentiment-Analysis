# Import libraries
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')
# -------------------- SETUP --------------------
st.set_page_config(
    page_title="Interactive Sentiment Classifier",
    page_icon="💬",
    layout="centered"
)
# -------------------- LOAD MODELS --------------------
# Models grouped by Vectorization technique

MODELS = {
    "Bag of Words (BOW)": {
        "Logistic Regression": {
            "model": joblib.load("trained_models/lr_bow.joblib"),
            "vectorizer": joblib.load("trained_models/bow_vectorizer.joblib"),
            "accuracy": 85.31,
            "desc": "Best generalization (Recommended) ⭐"
        },
        "SVM": {
            "model": joblib.load("trained_models/svm_bow.joblib"),
            "vectorizer": joblib.load("trained_models/bow_vectorizer.joblib"),
            "accuracy": 83.92,
            "desc": "Strong linear margin classifier"
        },
        "Multinomial Naive Bayes": {
            "model": joblib.load("trained_models/mnb_bow.joblib"),
            "vectorizer": joblib.load("trained_models/bow_vectorizer.joblib"),
            "accuracy": 82.52,
            "desc": "Fast probabilistic baseline"
        }
    },
    "TF-IDF": {
        "Multinomial Naive Bayes": {
            "model": joblib.load("trained_models/mnb_tf_idf.joblib"),
            "vectorizer": joblib.load("trained_models/tf_idf_vectorizer.joblib"),
            "accuracy": 85.31,
            "desc": "Best TF-IDF performer ⚡"
        },
        "SVM": {
            "model": joblib.load("trained_models/svm_tf_idf.joblib"),
            "vectorizer": joblib.load("trained_models/tf_idf_vectorizer.joblib"),
            "accuracy": 84.62,
            "desc": "Robust margin-based model"
        },
        "KNN": {
            "model": joblib.load("trained_models/knn_tf_idf.joblib"),
            "vectorizer": joblib.load("trained_models/tf_idf_vectorizer.joblib"),
            "accuracy": 81.82,
            "desc": "Distance-based, slower inference"
        }
    }
}
# -------------------- TEXT CLEANING --------------------
def clean_text(text: str) -> str:
    text = re.sub(r"(https?://|www\.)\S+", " ", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().strip()

    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    if 'not' in stop_words:
        stop_words.remove('not')

    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)
# -------------------- HEADER --------------------
st.markdown("""
<h1 style='text-align:center;'>💬 Interactive Sentiment Analyzer</h1>
<p style='text-align:center;'>Analyze emotions from text using Machine Learning</p>
""", unsafe_allow_html=True)
# -------------------- MODEL SELECTION --------------------
with st.expander("🧠 Choose Vectorization & Model", expanded=True):
    vec_type = st.radio(
        "Select feature extraction method",
        options=list(MODELS.keys()),
        horizontal=True
    )
    model_name = st.selectbox(
        "Select classification model",
        options=list(MODELS[vec_type].keys())
    )
model_info = MODELS[vec_type][model_name]
model = model_info['model']
vectorizer = model_info['vectorizer']

st.metric("Test Accuracy (%)", model_info['accuracy'])
st.caption(model_info['desc'])
st.divider()
# -------------------- USER INPUT --------------------
text = st.text_area(
    "✍️ Enter your text",
    placeholder="Type a sentence, review, or social media post...",
    height=150
)
hashtags = st.text_input(
    "#️⃣ Optional hashtags",
    placeholder="#happy #sad #excited"
)
full_text = text + " " + hashtags.replace('#', ' ')
# -------------------- ANALYZE BUTTON --------------------
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(full_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(vector).max() * 100

        st.divider()
        st.subheader("📊 Analysis Result")

        if prediction == 1:
            st.success("😊 **Positive Sentiment**")
            st.write("You're expressing positive emotions. Keep the good vibes going!")
        elif prediction == 0:
            st.info("😐 **Neutral Sentiment**")
            st.write("Your text sounds neutral. Not too emotional, not too dull.")
        else:
            st.error("😞 **Negative Sentiment**")
            st.write("Your text shows negative emotions. Take a moment to recharge.")

        if confidence is not None:
            st.progress(int(confidence))
            st.caption(f"Prediction confidence: {confidence:.2f}%")
# -------------------- FOOTER --------------------
st.divider()
st.caption("Built with ❤️ using Streamlit & Machine Learning")
