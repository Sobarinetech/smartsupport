import streamlit as st
import google.generativeai as genai
from google.cloud import speech
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
from google.oauth2 import service_account

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Initialize the Google Cloud Speech-to-Text client
credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_CLOUD_CREDENTIALS"])
speech_client = speech.SpeechClient(credentials=credentials)

# Simulated knowledge base (replace with real DB or search API)
knowledge_base = {
    "customer query about payment": "You can check your payment status by visiting the payments section in your account.",
    "refund policy": "Our refund policy allows for full refunds within 30 days of purchase.",
    "shipping details": "Shipping takes between 5-7 business days depending on your location.",
}

# Streamlit App UI
st.title("Ever AI with Live Transcription")
st.write("Use Generative AI to get responses based on your speech input.")

# Speech-to-Text Class to capture audio and transcribe it
class AudioTransformer(VideoTransformerBase):
    def __init__(self):
        self.client = speech.SpeechClient(credentials=credentials)
        self.frames = []
        self.transcription = ""
    
    def transform(self, frame):
        audio_data = frame.to_ndarray(format="int16")
        self.frames.append(audio_data)
        return frame
    
    def transcribe_audio(self):
        # Combine all frames into a single byte stream
        audio_data = np.concatenate(self.frames).tobytes()

        # Configure audio for transcription
        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        # Send audio to Google Cloud Speech-to-Text for transcription
        response = self.client.recognize(config=config, audio=audio)

        # Combine results into a transcription string
        self.transcription = " ".join([result.alternatives[0].transcript for result in response.results])

# Display transcription and response in the UI
webrtc_ctx = webrtc_streamer(key="audio", video_transformer_class=AudioTransformer, media_stream_constraints={"audio": True, "video": False})

# Display transcribed text when available
if webrtc_ctx.video_transformer:
    transcription = webrtc_ctx.video_transformer.transcription
    if transcription:
        st.write(f"Transcription: {transcription}")

        # Search the knowledge base for relevant responses
        matched_response = None
        for query, response in knowledge_base.items():
            if query.lower() in transcription.lower():
                matched_response = response
                break

        if matched_response:
            st.write("Suggested Response from Knowledge Base:")
            st.write(matched_response)
        else:
            st.write("No exact match found in the knowledge base.")

        # Pass transcribed text to the Gemini model for response generation if no match is found
        if not matched_response:
            if st.button("Generate AI Response"):
                try:
                    # Generate response using Google PaLM (Gemini)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt=transcription)

                    # Display the response in Streamlit
                    st.write("AI Response:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
