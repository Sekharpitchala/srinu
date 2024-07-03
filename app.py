from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, render_template
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

# Initialize the transformers summarization pipeline
summarizer = pipeline("summarization")

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize_text(input_text, max_length=300, min_length=50):
    summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf)[0, 1]
    return similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    original_text = request.form['text']
    
    if original_text:
        summarized_text = summarize_text(original_text)
        original_text_clean = preprocess_text(original_text)
        summarized_text_clean = preprocess_text(summarized_text)
        similarity_score = calculate_similarity(original_text_clean, summarized_text_clean)
        
        # Determine plot type from form input
        plot_type = request.form['plot_type']
        labels = ['Original Text', 'Summarized Text']
        scores = [1.0, similarity_score]
        plot_data = plot_similarity(labels, scores, plot_type)
        
        return render_template('result.html', 
                               original_text=original_text, 
                               summarized_text=summarized_text, 
                               similarity_score=similarity_score, 
                               plot_data=plot_data)
    else:
        return jsonify({'error': 'No text provided.'})

def plot_similarity(labels, scores, plot_type):
    x = np.arange(len(labels))
    
    if plot_type == 'bar':
        plt.bar(x, scores, color=['blue', 'orange'])
    elif plot_type == 'scatter':
        plt.scatter(x, scores, color='red')
    
    plt.xlabel('Text')
    plt.ylabel('Similarity Score')
    plt.title('Similarity between Original and Summarized Text')
    plt.xticks(x, labels)
    plt.ylim(0, 1.2)  # Adjust ylim if needed
    
    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return plot_data

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)
