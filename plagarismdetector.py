resumes_folder = '/content/drive/MyDrive/RESUME DETECTOR'

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.
    """
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
    """
    Calculates pairwise cosine similarity between resumes.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(resume_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def main(resumes_folder, plagiarism_threshold=0.8):
    # Extract resumes' text
    resume_texts = []
    resume_files = [file for file in os.listdir(resumes_folder) if file.endswith('.pdf')]
    for resume_file in resume_files:
        resume_path = os.path.join(resumes_folder, resume_file)
        text = extract_text_from_pdf(resume_path)
        if text:
            resume_texts.append((resume_file, text))
        else:
            print(f"Failed to extract text from {resume_file}")

    if len(resume_texts) < 2:
        print("Not enough resumes for plagiarism detection.")
        return

    # Calculate similarity matrix
    filenames, texts = zip(*resume_texts)
    similarity_matrix = calculate_similarity_matrix(texts)

    # Check for plagiarism
    print("\nPlagiarism Results:")
    flagged_pairs = []
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            similarity_score = similarity_matrix[i][j]
            if similarity_score >= plagiarism_threshold:
                flagged_pairs.append((filenames[i], filenames[j], similarity_score))
                print(f"Plagiarism Detected: {filenames[i]} and {filenames[j]} (Score: {similarity_score:.2f})")

    if not flagged_pairs:
        print("No plagiarism detected.")

resumes_folder = "/content/drive/MyDrive/RESUME DETECTOR"  # Folder containing resumes as PDFs
plagiarism_threshold = 0.8                # Threshold for plagiarism detection (0 to 1)
main(resumes_folder, plagiarism_threshold)