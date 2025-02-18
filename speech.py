import speech_recognition as sr
import pyttsx3

class SpeechInterface:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError:
            print("API request error.")
            return None

    def speak(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
