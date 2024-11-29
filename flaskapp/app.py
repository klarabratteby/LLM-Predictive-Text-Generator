from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from textstat import flesch_reading_ease
import torch

app = Flask(__name__)

# Load the original BART model
model_id = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_id)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Load the fine-tuned BART model
fine_tuned_model_id = "../results/bart_finetune_cnn"  # Replace with the correct path if necessary
try:
    fine_tuned_bart = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_id)
    print("Fine-tuned model loaded successfully.")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    fine_tuned_bart = None  # Set to None if the model fails to load

def generate_continuation(model, text, max_output_length=150):
    """
    Generates a continuation of the input text using the specified model.
    """
    # Encode the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    
    # Generate continuation
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=len(inputs['input_ids'][0]) + max_output_length,  # Control length of generated text
        num_beams=5,  # Beam search for better quality
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return generated_text

def calculate_readability(summary):
    """
    Calculates the Flesch Reading Ease score and assigns a readability description.
    """
    score = flesch_reading_ease(summary)
    description = readability_description(score)
    return {
        "flesch_reading_ease": score,
        "description": description
    }

def readability_description(score):
    """Assigns a readability description based on Flesch Reading Ease score."""
    if score >= 90:
        return "Very Easy"
    elif score >= 80:
        return "Easy"
    elif score >= 70:
        return "Fairly Easy"
    elif score >= 60:
        return "Standard"
    elif score >= 50:
        return "Fairly Difficult"
    elif score >= 30:
        return "Difficult"
    else:
        return "Very Confusing"

def calculate_semantic_redundancy(summary):
    """
    Calculates semantic redundancy as the proportion of unique words in the summary
    and assigns a description based on the score.
    """
    words = summary.split()
    unique_words = set(words)
    redundancy_score = len(unique_words) / len(words) if words else 0
    description = redundancy_description(redundancy_score)
    return {"score": round(redundancy_score, 2), "description": description}  # Return both score and description

def redundancy_description(score):
    """Assigns a description based on the semantic redundancy score."""
    if score >= 0.9:
        return "Amazing vocabulary"
    elif score >= 0.75:
        return "Good vocabulary"
    elif score >= 0.5:
        return "Average vocabulary"
    else:
        return "Bad vocabulary"



@app.route("/", methods=["GET", "POST"])
def home():
    original_continuation = ""
    fine_tuned_continuation = ""
    original_readability = {}
    fine_tuned_readability = {}
    original_redundancy = {}
    fine_tuned_redundancy = {}

    if request.method == "POST":
        text = request.form["text"]

        if text:
            # Generate continuation using the original BART model
            original_continuation = generate_continuation(bart_model, text)
            original_readability = calculate_readability(original_continuation)
            original_redundancy = calculate_semantic_redundancy(original_continuation)

            # Generate continuation using the fine-tuned BART model, if loaded
            if fine_tuned_bart:
                fine_tuned_continuation = generate_continuation(fine_tuned_bart, text)
                fine_tuned_readability = calculate_readability(fine_tuned_continuation)
                fine_tuned_redundancy = calculate_semantic_redundancy(fine_tuned_continuation)
            else:
                fine_tuned_continuation = "Fine-tuned model could not be loaded."
                fine_tuned_readability = {"flesch_reading_ease": "N/A", "description": "N/A"}
                fine_tuned_redundancy = "N/A"

    return render_template("index.html",
                           original_prediction=original_continuation,
                           fine_tuned_prediction=fine_tuned_continuation,
                           original_readability=original_readability,
                           fine_tuned_readability=fine_tuned_readability,
                           original_redundancy=original_redundancy,
                           fine_tuned_redundancy=fine_tuned_redundancy,
                           text=request.form.get("text", ""))

if __name__ == "__main__":
    app.run(debug=True)