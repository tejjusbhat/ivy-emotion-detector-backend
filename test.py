import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("fine-tuned-tinybert")
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-tinybert")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def predict_emotion(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    prediction = torch.argmax(logits, dim=-1).item()
    label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    predicted_label = label_names[prediction]
    
    return predicted_label

