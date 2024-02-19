# PDF Story Summarizer & Text-to-Audio Converter

This is a web application that takes a PDF containing a story as input, generates a summary using the Llama2 model from Ollama, and provides an audio output of the summary.

## Features

- Upload a PDF file containing a story.
- Extract text from the PDF and generate a summary using the Llama2 model.
- Convert the summary into an audio file using Text-to-Speech (TTS) technology.
- Listen to the audio output directly within the application.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/pdf-story-summarizer.git
```

2. Navigate to the project directory:
```bash
cd pdf-story-summarizer
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama for LLM:

- [Ollama](https://ollama.com/)

5. Open terminal and run:
```bash
ollama run llama2
```

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

- Access the application in your web browser at http://localhost:8501.
- Upload a PDF file containing a story.
- Wait for the summary generation and audio conversion process to complete.
- Listen to the audio output by playing the provided audio file (story.wav).

## Credits
This application was created by YATIN KUNDRA.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
