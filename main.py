from dotenv import load_dotenv
from emotion_detector import EmotionDetector
from chat_handler import ChatHandler
from speech import SpeechInterface

class VoiceAssistant:
    def __init__(self):
        load_dotenv()
        self.emotion_detector = EmotionDetector()
        self.chat_handler = ChatHandler()
        self.speech_interface = SpeechInterface()
        self.listening = True

    def run_voice_assistant(self):
        user_input = self.speech_interface.listen()
        if not user_input:
            return
        emotion = self.emotion_detector.predict(user_input)
        response = self.chat_handler.get_response(user_input, emotion)
        self.speech_interface.speak(response)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run_voice_assistant()
