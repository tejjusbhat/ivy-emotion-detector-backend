import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class EmotionDetector:
    def __init__(self, model_name="fine-tuned-tinybert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
        return self.labels[prediction]
