import gradio as gr
import spacy
from transformers import pipeline
import openai
import os

# Load Spacy Model
#nlp = spacy.load("en_core_web_sm")

# Initialize HuggingFace sentiment analysis pipeline
#sentiment_analysis = pipeline("sentiment-analysis")

# Set OpenAI API key
openai.api_key = os.environ['key3']


# Initialize HuggingFace pipelines
#nlp_ner = pipeline("ner", model="dslim/bert-base-NER")
#nlp_ner = pipeline("ner", model="flair/ner-english-fast")

#nlp_sentiment = pipeline("sentiment-analysis")


# Define function for Gradio Interface
def process_text(text):
   
    # Grammar correction
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{text}\n\nCorrected version:",
        temperature=0.3,
        max_tokens=100
    )
    corrected_text = response.choices[0].text.strip()

    return  corrected_text

# Define Gradio Interface
iface = gr.Interface(
    fn=process_text,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter text here..."),
    outputs=[
             gr.outputs.Textbox(label="Corrected Text")],
    examples=[
        ["Apple is planings to open a nw tore in San Francisco on january 1, 2024."],
        ["I reallly wants to run."],
        ["The quck bown fox jumps over the azy dog."]
    ],
    title="Grammarly-like App",
    description="A simple 'Grammarly'-like app OpenAI GPT-3. It corrects grammatical errors."
)

iface.launch()
