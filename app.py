import streamlit as st
import openai
import requests
import json
from moviepy.editor import VideoFileClip, AudioFileClip
from google.cloud import speech, texttospeech
import tempfile
import ffmpeg
import os

def main():
    st.title("Generative Audio")
    
    upload_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi"])
    
    if upload_file is not None:
        # Create a temporary file for the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(upload_file.read())
            temp_video_path = temp_video.name  # Get the file path
        
        # Load the video from the temporary file
        video_clip = VideoFileClip(temp_video_path)
        
        # Create another temporary file for the extracted audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_file = temp_audio.name  # Get the file path for the audio
            video_clip.audio.write_audiofile(audio_file, codec='pcm_s16le')
        
        st.info("Audio Extracted.....")
    
        # Transcribe the audio (you can implement your own function)
        transcription = transcribe_audio(audio_file)
        st.write("Transcription:", transcription)
        
        st.info("Audio Transcription done....")
        
        # Correct the transcription (you can implement your own function)
        corrected_transcription = correct_transcription(transcription)
        st.write("Corrected Transcription:", corrected_transcription)
        
        st.info("Audio Corrected done....")
        
        # Generate new audio from corrected transcription (you can implement your own function)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as generated_audio:
            generated_audio_file = generated_audio.name  # Get path for generated audio
            generate_audio(corrected_transcription, generated_audio_file)
        
        st.audio(generated_audio_file, format='audio/wav')
        
        # Replace the audio in the original video (you can implement your own function)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as final_video:
            final_video_path = final_video.name  # Get path for final video
            replace_audio_in_video(temp_video_path, generated_audio_file, final_video_path)
        
        st.video(final_video_path)

def transcribe_audio(audio_file):
    client = speech.SpeechClient()
    with open(audio_file, "rb") as af:
        audio_data = af.read()
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    response = client.recognize(config=config, audio=audio)
    transcription = ' '.join([result.alternatives[0].transcript for result in response.results])
    return transcription

def correct_transcription(transcription):
    azure_openai_key = "22ec84421ec24230a3638d1b51e3a7dc"
    azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_openai_key
    }
    
    data = {
        "messages": [{"role": "user", "content" : f"Please correct this transcription: {transcription}"}],
        "max_tokens": 500
    }
    
    response = requests.post(azure_openai_endpoint, headers=headers, json=data)
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

def generate_audio(text):
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-J",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    
    audio_file = "corrected_audio.wav"
    
    with open(audio_file, 'wb') as out:
        out.write(response.audio_content)
    return audio_file

def replace_audio_in_video(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_video_path = "final_video.mp4"
    final_clip.write_videofile(final_video_path, codec='libx264', audio_codec='aac')
    
    return final_video_path

if __name__ == "__main__":
    main()
