from flask import Flask, request, jsonify, render_template
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import tempfile
import shutil
import zipfile
from collections import defaultdict
import re

# Create the Flask app object
app = Flask(__name__)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def calculate_similarity_matrix(resume_texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(resume_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix, vectorizer

import re

def find_common_text(text1, text2):
    """
    Finds common words between two texts, ignoring case and punctuation.
    """
    # Remove punctuation and convert to lowercase
    text1_cleaned = re.sub(r'[^\w\s]', '', text1.lower())
    text2_cleaned = re.sub(r'[^\w\s]', '', text2.lower())

    # Split into words
    words1 = set(text1_cleaned.split())
    words2 = set(text2_cleaned.split())

    # Find common words
    common_words = words1.intersection(words2)
    return common_words

def find_common_words_positions(text, common_words):
    positions = []
    for word in common_words:
        for match in re.finditer(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
            positions.append((match.start(), match.end()))
    return positions

@app.route('/')
def home():
    return render_template("page1.html")
@app.route('/resume', methods=['GET', 'POST'])
def handle_resume_upload():
    if request.method == 'GET':
        return render_template('upload.html')

    if request.method == 'POST':
        temp_file = None
        extracted_folder = None
        try:
            if 'file' not in request.files:
                return render_template('results.html', error='No file uploaded')

            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return render_template('results.html', error='No file selected')

            # Check file size
            MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
            uploaded_file.seek(0, os.SEEK_END)
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)
            if file_size > MAX_FILE_SIZE:
                return render_template('results.html', error='File size exceeds the limit of 10 MB')

            # Save the file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                uploaded_file.save(temp_file.name)

            # Validate ZIP file
            if not zipfile.is_zipfile(temp_file.name):
                return render_template('results.html', error='Uploaded file is not a valid ZIP file')

            # Extract and process files
            extracted_folder = tempfile.mkdtemp()
            with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                zip_ref.extractall(extracted_folder)

            print(f"Extracted folder: {extracted_folder}")  # Debug log
            print(f"Contents of extracted folder: {os.listdir(extracted_folder)}")  # Debug log

            # Extract text from all PDF files in the zip
            resume_texts = []
            filenames = []
            file_texts = {}
            for root, _, files in os.walk(extracted_folder):
                for file in files:
                    if file.endswith('.pdf'):
                        file_path = os.path.join(root, file)
                        print(f"Processing file: {file_path}")  # Debug log
                        text = extract_text_from_pdf(file_path)
                        if not text:
                            print(f"Warning: Could not extract text from {file}. Skipping.")
                            continue
                        print(f"Extracted text length for {file}: {len(text)}")  # Debug log
                        resume_texts.append(text)
                        filenames.append(file)
                        file_texts[file] = text

            print(f"Total valid resumes: {len(resume_texts)}")  # Debug log

            if len(resume_texts) < 2:
                return render_template('results.html', error='Not enough resumes for plagiarism detection')

            # Calculate similarity
            similarity_matrix, vectorizer = calculate_similarity_matrix(resume_texts)
            plagiarism_threshold = 0.3

            # Check for plagiarism
            flagged_pairs = []
            for i in range(len(filenames)):
                for j in range(i + 1, len(filenames)):
                    similarity_score = similarity_matrix[i][j]
                    print(f"Similarity score between {filenames[i]} and {filenames[j]}: {similarity_score}")  # Debug log
                    if similarity_score >= plagiarism_threshold:
                        common_words = find_common_text(resume_texts[i], resume_texts[j])
                        flagged_pairs.append({
                            'file1': filenames[i],
                            'file2': filenames[j],
                            'score': round(similarity_score, 2),
                            'common_text': list(common_words),
                            'file1_text': file_texts[filenames[i]],
                            'file2_text': file_texts[filenames[j]],
                        })

            return render_template('results.html', results=flagged_pairs)

        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
            return render_template('results.html', error=f'Error processing file: {str(e)}')
        finally:
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            if extracted_folder and os.path.exists(extracted_folder):
                shutil.rmtree(extracted_folder)
if __name__ == '__main__':
    app.run(debug=True)