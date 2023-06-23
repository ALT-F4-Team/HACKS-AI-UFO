import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CustomTextClassifier:
    def __init__(self, model_path, tokenizer_path, n_classes=2, model_save_path='/content/best.pt'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = 512

        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)

        self.model.to(self.device)

    def predict(self, text, ind_to_labels):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return ind_to_labels[prediction]