from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import re
import nltk
import psycopg2
import yt_dlp  # Added for downloading audio
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable, NoTranscriptFound
from google.cloud import speech
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from loguru import logger
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # Restrict CORS for security

# Load NLP model
logger.info("Loading NLP model...")
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    logger.info("✅ NLP Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ NLP Model failed to load: {e}")

# Ensure NLTK is properly set up
nltk.download('punkt')

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "smart_notes"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def extract_video_id(url):
    """Extracts video ID from various YouTube URL formats."""
    patterns = [
        r"youtu\.be/([0-9A-Za-z_-]{11})",      # Shortened URL
        r"youtube\.com/watch\?v=([0-9A-Za-z_-]{11})",  # Standard URL
        r"youtube\.com/embed/([0-9A-Za-z_-]{11})",     # Embedded URL
        r"youtube\.com/v/([0-9A-Za-z_-]{11})"         # Old URL format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    logger.error("❌ Could not extract video ID from URL")
    return None


def get_youtube_transcript(video_url):
    """Extract subtitles from YouTube video."""
    video_id = extract_video_id(video_url)
    if not video_id:
        logger.error("Invalid YouTube URL")
        return None
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])

    except TranscriptsDisabled:
        logger.warning("❌ No subtitles available for this video.")
        return None

    except NoTranscriptFound:
        logger.warning("❌ No transcript found for this video.")
        return None

    except VideoUnavailable:
        logger.warning("❌ Video is unavailable or restricted.")
        return None

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def download_audio(video_url):
    """Download YouTube video audio using yt-dlp and verify success."""
    output_filename = "temp_audio.wav"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False  # Show logs for debugging
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if not os.path.exists(output_filename):  # Check if file was created
            logger.error("❌ Audio file was not downloaded!")
            return None

        logger.info("✅ Audio downloaded successfully!")
        return output_filename

    except Exception as e:
        logger.error(f"❌ Audio download failed: {e}")
        return None


def convert_speech_to_text(audio_file):
    """Convert speech to text using Google Speech-to-Text API and debug output."""
    try:
        client = speech.SpeechClient()
        with open(audio_file, "rb") as audio:
            content = audio.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,  # Ensure correct sample rate
            language_code="en-US"
        )

        response = client.recognize(config=config, audio=audio)
        
        if not response.results:
            logger.error("❌ No speech detected in the audio!")
            return None
        
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        logger.info(f"✅ Speech-to-Text Output: {transcript}")
        return transcript

    except Exception as e:
        logger.error(f"❌ Speech-to-Text failed: {e}")
        return None


def get_transcript_or_audio(video_url):
    """Extract subtitles or use audio if subtitles are unavailable."""
    video_id = extract_video_id(video_url)
    if not video_id:
        logger.error("❌ Invalid YouTube URL")
        return None

    logger.info("🔍 Trying to fetch subtitles...")
    transcript = get_youtube_transcript(video_url)

    if transcript:
        logger.info("✅ Subtitles found, using transcript.")
        return transcript

    logger.warning("⚠️ No subtitles found. Extracting speech from audio...")

    audio_path = download_audio(video_url)
    if not audio_path:
        logger.error("❌ Failed to download audio!")
        return None

    logger.info("🎙️ Converting speech to text...")
    transcript = convert_speech_to_text(audio_path)

    os.remove(audio_path)  # Cleanup

    if not transcript:
        logger.error("❌ No speech detected in the video.")
        return None

    logger.info("✅ Successfully extracted speech.")
    return transcript


def chunk_text(text, chunk_size=1000):
    """Splits text into smaller chunks to avoid token limits."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize_text(text):
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = []

    for chunk in chunks:
        if len(chunk.strip()) == 0:
            continue  

        try:
            summary = summarizer(chunk, max_length=80, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Summarization error: {e}")

    structured_notes = f"""
    ### Introduction:
    {summaries[0] if summaries else "No introduction available."}

    ### Key Points:
    - **Point 1:** {summaries[1] if len(summaries) > 1 else "No key points found."}
    - **Point 2:** {summaries[2] if len(summaries) > 2 else "No additional key points found."}
    - **Point 3:** {summaries[3] if len(summaries) > 3 else "No more key points available."}

    ### Summary:
    {" ".join(summaries[:3])}  

    ### Conclusion:
    {summaries[-1] if summaries else "No conclusion available."}
    """

    return structured_notes.strip()





def save_to_database(video_url, notes):
    """Save summarized notes to PostgreSQL while avoiding duplicates."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO notes (video_url, summary) 
            VALUES (%s, %s) 
            ON CONFLICT (video_url) DO UPDATE 
            SET summary = EXCLUDED.summary
        """, (video_url, notes))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Data saved to database successfully!")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")


def generate_pdf(notes, filename="static/output.pdf"):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()

        heading_style = ParagraphStyle(name="Heading", fontSize=14, spaceAfter=12, bold=True)
        normal_style = ParagraphStyle(name="Normal", fontSize=12, spaceAfter=10)
        bullet_style = ParagraphStyle(name="Bullet", fontSize=12, spaceAfter=8, leftIndent=20)

        story = []

        sections = notes.split("\n\n")
        for section in sections:
            if section.startswith("### "):  
                story.append(Spacer(1, 12))  
                story.append(Paragraph(section.replace("### ", ""), heading_style))
                story.append(Spacer(1, 8))  
            elif section.startswith("- "):  
                story.append(Paragraph(section, bullet_style))
            else:  
                story.append(Paragraph(section, normal_style))
            
            story.append(Spacer(1, 8))  

        doc.build(story)
        print("✅ PDF structured successfully!")
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")



@app.route("/")
def home():
    return "Smart Learning Notes API is running!"

@app.route("/process", methods=["POST"])
def process_video():
    data = request.json
    logger.info(f"Received request: {data}")  # Debugging

    video_url = data.get("video_url")
    if not video_url:
        return jsonify({"error": "❌ Missing YouTube URL"}), 400

    logger.info("Fetching transcript...")
    transcript = get_transcript_or_audio(video_url)  # Updated function call

    if transcript is None:
        return jsonify({"error": "❌ No subtitles or speech detected."}), 400

    logger.info("Summarizing text...")
    summarized_text = summarize_text(transcript)

    logger.info("Saving to database...")
    save_to_database(video_url, summarized_text)

    logger.info("Generating PDF...")
    generate_pdf(summarized_text)

    logger.info("✅ Process completed!")
    return jsonify({"summary": summarized_text, "pdf_url": "http://127.0.0.1:5001/static/output.pdf"})

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
