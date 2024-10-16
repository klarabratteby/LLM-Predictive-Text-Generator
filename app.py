from flask import Flask, render_template, request, redirect, url_for
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
import os
import PyPDF2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('results/gpt2_finetuned')
model = GPT2LMHeadModel.from_pretrained('results/gpt2_finetuned')

# Summarization model (e.g., T5 for better summarization)
summarizer_model = T5ForConditionalGeneration.from_pretrained('t5-small')
summarizer_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text


def extract_text_from_word(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def summarize_document(document_text, summary_type="bullet", detail_level="short", tone="formal"):
    if summary_type == "bullet":
        # Custom bullet point summary
        prompt = f"Summarize the following document in bullet points: {document_text}"
    elif summary_type == "abstract":
        prompt = f"Generate an abstract of the following document: {document_text}"

    if detail_level == "detailed":
        prompt += " Provide a detailed summary."
    else:
        prompt += " Provide a short summary."

    if tone == "casual":
        prompt += " Make the tone casual."
    else:
        prompt += " Keep the tone formal."

    inputs = summarizer_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = summarizer_model.generate(inputs['input_ids'], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Determine file type and extract text
        if file.filename.endswith('.pdf'):
            document_text = extract_text_from_pdf(file_path)
        elif file.filename.endswith('.docx'):
            document_text = extract_text_from_word(file_path)
        else:  # Assuming text file
            document_text = extract_text_from_txt(file_path)

        # Custom summarization options
        summary_type = request.form.get('summary_type', 'bullet')
        detail_level = request.form.get('detail_level', 'short')
        tone = request.form.get('tone', 'formal')

        summary = summarize_document(document_text, summary_type, detail_level, tone)
        
        return render_template('result.html', summary=summary)


# Route to handle chat interaction
@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        generated_text = generate_text(user_input)
        return render_template('chat.html', generated_text=generated_text)

def generate_text(prompt, max_length=300):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(
        inputs['input_ids'], 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    app.run(debug=True)
