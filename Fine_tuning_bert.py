from transformers import BertTokenizer, BertForQuestionAnswering
import torch


class SimpleBERTQA:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)


    def answer_question(self, context, question):
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)

        outputs = self.model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        answer = ' '.join(tokens[answer_start:answer_end + 1])

        answer = answer.replace(' ##', '').replace('[CLS]', '').replace('[SEP]', '').strip()

        return answer

    def fine_tune(self, train_contexts, train_questions, train_answers, epochs=3, batch_size=2):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(epochs):
            total_loss = 0

            for i in range(0, len(train_contexts), batch_size):
                batch_contexts = train_contexts[i:i + batch_size]
                batch_questions = train_questions[i:i + batch_size]
                batch_answers = train_answers[i:i + batch_size]

                inputs = self.tokenizer.encode_plus(
                    batch_questions[0],
                    batch_contexts[0],
                    return_tensors='pt',
                    max_length=512,
                    truncation=True
                ).to(self.device)


                answer_start = torch.tensor([batch_answers[0]['start']], device=self.device)
                answer_end = torch.tensor([batch_answers[0]['end']], device=self.device)


                outputs = self.model(**inputs,
                                     start_positions=answer_start,
                                     end_positions=answer_end)

                loss = outputs.loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")


def main():
    qa = SimpleBERTQA()
    context = "The Python programming language was created by Guido van Rossum and released in 1991."
    question = "Who created Python?"

    answer = qa.answer_question(context, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    train_contexts = [
        "BERT was developed by Google in 2018.",
        "PyTorch was created by Facebook's AI Research lab."
    ]
    train_questions = [
        "Who developed BERT?",
        "Who created PyTorch?"
    ]
    train_answers = [
        {'start': 4, 'end': 6},
        {'start': 4, 'end': 7}
    ]

    qa.fine_tune(train_contexts, train_questions, train_answers, epochs=1)


main()