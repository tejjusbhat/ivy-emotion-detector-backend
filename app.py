from dotenv import load_dotenv
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import texttospeech, speech
from pydub import AudioSegment
from pydub.playback import play
import io

load_dotenv()

emotion_classification_model = AutoModelForSequenceClassification.from_pretrained("fine-tuned-tinybert")
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-tinybert")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
emotion_classification_model.to(device)

gpt_model = ChatGoogleGenerativeAI(model="gemini-pro")
output_parser = StrOutputParser()
prompt_template = """You are a personal assistant, your job is to help the user with anything they ask.
                    Keep your answers short and to the point. As if you are speaking to a friend.
                    Your response should be to make the user happy, the user's current emotion is {emotion}.
                    The user's statement is as follows:
                    {user_prompt}"""
prompt = ChatPromptTemplate.from_template(prompt_template)

tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()

def predict_emotion(text: str) -> str:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    emotion_classification_model.eval()

    with torch.no_grad():
        outputs = emotion_classification_model(**inputs)
        logits = outputs.logits

    prediction = torch.argmax(logits, dim=-1).item()
    label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    predicted_label = label_names[prediction]

    return predicted_label

def text_to_speech(text: str):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    
    audio_data = io.BytesIO(response.audio_content)
    audio_segment = AudioSegment.from_file(audio_data, format="mp3")
    play(audio_segment)

def speech_to_text() -> str:
    audio_file = "sad.wav"
    
    with io.open(audio_file, "rb") as audio:
        content = audio.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, language_code="en-US")
    
    response = stt_client.recognize(config=config, audio=audio)
    
    for result in response.results:
        return result.alternatives[0].transcript
    
    return ""

def chatbot():
    print("Listening...")
    user_input = speech_to_text()
    if user_input:
        print(f"User said: {user_input}")
        
        emotion = predict_emotion(user_input)
        print("The predicted emotion is:", emotion)
        
        chain = prompt | gpt_model | output_parser
        ai_response = chain.invoke({"emotion": emotion, "user_prompt": user_input})
        
        print(f"Bot response: {ai_response}")
        
        text_to_speech(ai_response)

if __name__ == "__main__":
    chatbot()
