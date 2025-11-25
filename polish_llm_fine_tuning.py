import pandas as pd
from transformers import EarlyStoppingCallback
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer
from datasets import Dataset
from google.colab import files
from transformers import TrainingArguments, Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification



df = pd.read_pickle("df.pkl") # wczytanie danych

df['labels'] = df['intencja'].astype('category').cat.codes # liczby zamiast etykiet
label_map = dict(enumerate(df['intencja'].astype('category').cat.categories)) # mapa etykiet

# train test val split
dataset = Dataset.from_pandas(df[['tekst', 'labels']])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_valid = dataset["train"].train_test_split(test_size=0.1, seed=42)

dataset["train"] = train_valid["train"]
dataset["val"] = train_valid["test"]


model_name = "allegro/herbert-base-cased" # polski model llm
tokenizer = AutoTokenizer.from_pretrained(model_name) # tokenizer do modelu

def tokenize_function(example): # tokenizacja
    return tokenizer(example["tekst"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# setup modelu
num_labels = len(df['labels'].unique()) # liczba klas
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def compute_metrics(eval_pred): # obliczanie metryk
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1) # zmiana logitow na indeksy klas z najwyzszym prawdopodobienstwem
    acc = accuracy_score(labels, preds) # accuracy
    f1 = f1_score(labels, preds, average='weighted') # f1
    return {"accuracy": acc, "f1": f1}

# parametry treningowe
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", # ewaluacja po kazdej epoce
    save_strategy="epoch", # zapisywanie modelu po kazdej epoce
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs", # katalog na logi treningu
    load_best_model_at_end=True, # po treningu wczytaj najlepszy model
    metric_for_best_model="f1", # wybor najlepszego modelu po f1
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # zatrzymanie po 2 epokach bez poprawy
)

trainer.train() # trenowanie modelu

trainer.save_model("evaluation-classifier-herbert") # zapis modelu


# ewaluacja
print("\n Ewaluacja na zbiorze testowym:")
metrics = trainer.evaluate(tokenized_dataset["test"])
print(metrics)


preds = trainer.predict(tokenized_dataset["test"])


# tworzenie confusion matrix

y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title(" Macierz pomyłek")
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa etykieta")

for text in disp.text_.ravel():
    text.set_text(f"{float(text.get_text()):.1f}")

plt.tight_layout()
plt.show()