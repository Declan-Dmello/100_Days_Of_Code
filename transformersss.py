import torch
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from typing import List , Tuple


class Transformerr:
    def __init__(self, model_name :  str ="distilbert-base-uncased-finetuned-sst-2-english") -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()


    def analyze_sentiments(self, text:str) -> Tuple[str,float]:
        try:
            inputs = self.tokenizer(text,return_tensors="pt", truncation = True, max_length=512)
            inputs = {name: tensor.to(self.device) for name , tensor in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits,dim=1)

            prediction = torch.argmax(probabilities).item()
            confidence = torch.max(probabilities).item()

            sentiment = "Positive" if prediction ==1 else "Negative"

            return sentiment , confidence

        except Exception as e:
            print(f"Error during Sentiment Analysis{str(e)}")
            return "error",0.0

    def batch_analyze(self,texts: List[str]) -> List[Tuple[str,float]]:
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                    truncation=True,max_length=512)
            inputs = {name: tensor.to(self.device) for name , tensor in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)

            results =[]
            for probs in probabilities:
                prediction = torch.argmax(probs).item()
                confidence = torch.max(probs).item()
                sentiment = "Positive" if prediction ==1 else "Negative"
                results.append((sentiment, confidence))
            return results

        except Exception as e :
            print(f"Error during batch analysis : {str(e)}")
            return [(f"error ",0.0) * len(texts)]


def main():
    transformer = Transformerr()

    text = "Harry Potter is the best book series."
    sentiment, confidence = transformer.analyze_sentiments(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")

    texts = [
        "The Sound of Music is an Amazing Musical",
        "The Earth is facing the wrath of human pollution",
        "The food was not that good"
    ]
    results = transformer.batch_analyze(texts)

    print("\nMultiple Sentence Analysis : ")
    for text, (sentiment, confidence) in zip(texts, results):
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")

main()
