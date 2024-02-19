import pdfplumber
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


st.header("Long Story Short")

pdf_file = st.file_uploader("Upload your pdf", type=["pdf"])


def extract_text_from_pdf(pdf_file):
    if pdf_file is not None:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    else:
        return "No PDF file uploaded."


def generate_summary(text):
    llm = Ollama(model="llama2")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world-class story teller."),
        ("user", f"tell me this story so that i know what happened in detail,  story : {text}  ")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    summary = chain.invoke({"input": text})
    return summary


from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

def tts(message):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Split the input text into smaller chunks
    chunk_size = 512  # You can adjust this value based on your model's maximum input length
    chunks = [message[i:i+chunk_size] for i in range(0, len(message), chunk_size)]

    speech = None

    for chunk in chunks:
        inputs = processor(text=chunk, return_tensors="pt")

        # Ensure the input sequence length does not exceed the model's maximum sequence length
        max_seq_length = model.config.max_length
        if inputs["input_ids"].shape[1] > max_seq_length:
            inputs["input_ids"] = inputs["input_ids"][:, :max_seq_length]

        # Load xvector containing speaker's voice characteristics from a dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        chunk_speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        if speech is None:
            speech = chunk_speech
        else:
            # Concatenate the speech chunks
            speech = torch.cat((speech, chunk_speech), dim=-1)

    sf.write("story.wav", speech.numpy(), samplerate=16000)
    return speech


if pdf_file is not None:
    extracted_text = extract_text_from_pdf(pdf_file)
    st.write("Extracted Text:")
    st.write(extracted_text)

    print("Generating Summary...")
    summary = generate_summary(extracted_text)
    st.write("Summary:")
    st.write(summary)
    st.write("this was the summary")

    print("Generating Audio...")
    tts(summary)

    audio_bytes = open("story.wav", "rb").read()
    st.audio(audio_bytes, format="audio/wav")
