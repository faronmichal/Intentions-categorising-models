from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import re
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import fasttext
import fasttext.util


# api do open ai
client = OpenAI(api_key="kod api")

# kategorie do których generujemy dane,
# kategorie są kategoriami z eccomerce, jest to zestaw który mógłby 
# wystąpić w praktyce, kategorie są przygotowane tak żeby miały sens
# z biznesowego punktu widzenia jednocześnie nie nachodząc na siebie
# nawzajem w celu umożliwienia sensownego modelowania i predykcji
categories = [
    "Złożenie zamówienia",
    "Anulowanie zamówienia",
    "Status przesyłki",
    "Zwrot lub reklamacja",
    "Płatność i faktura",
    "Dostępność produktu",
    "Promocje i rabaty",
    "Zmiana danych zamówienia",
    "Problem techniczny",
    "Opinie i sugestie",
    "Rejestracja i logowanie",
    "Program lojalnościowy i punkty"
]

# parametry do generowania danych
examples_per_batch = 40   # ile przykladow generujemy na batch
batches_per_category = 15   # liczba batchy
model_name = "gpt-5-mini"  # model
temperature = 1.0   # losowość generowanego tekstu        
max_tokens = 600   # maksymalna liczba tokenów           
top_p = 0.9  # nucleus sampling, bardziej precyzyjne odpowiedzi                 
frequency_penalty = 0.3   # kara za powtarzanie słów    
presence_penalty = 0.4  # zachęta do używania nowych słów

data = []

# generowanie danych, for loop dla kazdej kategorii
for i, category in enumerate(categories, 1):
    print(f"\n Kategoria {i}/{len(categories)}: {category}")

    for batch in range(1, batches_per_category + 1):
        print(f"   Batch {batch}/{batches_per_category} ({examples_per_batch} przykładów)")
        # prompt do chata generujący dane w odpowiednim formacie
        prompt = f"""
        Wygeneruj {examples_per_batch} różnych, realistycznych wypowiedzi klientów sklepu internetowego w języku polskim,
        które pasują do kategorii: "{category}".

        Każda wypowiedź powinna brzmieć naturalnie, jak prawdziwa wiadomość użytkownika do czatu obsługi klienta.

        Format odpowiedzi:
        Zwróć wynik w postaci listy obiektów JSON, gdzie każdy obiekt ma DOKŁADNIE dwa pola:
        - "tekst" – pojedyncze zdanie z wypowiedzią użytkownika (nie lista!),
        - "intencja" – zawsze dokładnie wartość "{category}".

        Przykład poprawnego formatu:
        [
          {{"tekst": "Chciałbym zwrócić buty, bo są za małe", "intencja": "Zwrot towaru"}},
          {{"tekst": "Proszę o etykietę zwrotną", "intencja": "Zwrot towaru"}}
        ]

        Nie używaj wypunktowań ani myślników przed tekstami.
        Zwróć tylko czysty JSON – bez żadnego komentarza, opisu, ani tekstu wokół.
        """

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            content = response.choices[0].message.content.strip()

            # wytnij sam json
            start = content.find("[")
            end = content.rfind("]") + 1
            clean_json = content[start:end]

            examples = json.loads(clean_json)
            print(f"   Otrzymano {len(examples)} przykładów")

            for ex in examples:
                if isinstance(ex, dict):
                    text = str(ex.get("tekst", "")).strip()
                    intent = str(ex.get("intencja", category)).strip()
                else:
                    text = str(ex).strip()
                    intent = category
                if text:
                    data.append({"tekst": text, "intencja": intent})

        except Exception as e:
            print(f"   Błąd w batchu {batch}: {e}")
            continue

        # pauza między requestami dla nie zawieszenia generowania
        time.sleep(1)

# format do dataframe
df = pd.DataFrame(data)

# czyszczenie końcówek jeśli model dodał niepotrzebne elementy
df["tekst"] = df["tekst"].astype(str)
df["tekst"] = df["tekst"].apply(
    lambda x: x.removeprefix("{'tekst': '").removesuffix("'}").strip()
)

# usuwanie pustych wierszy
df = df[df["tekst"].str.len() > 0]

# zapis
df.to_csv("intencje_ecommerce.csv", index=False, encoding="utf-8")

# informacje o wygenerowanych danych
print(f"łącznie rekordów: {len(df)}")
print(df.sample(10))
df.head(50)
df['intencja'].value_counts()


# usunięcie duplikatów
duplicates = df['tekst'].duplicated(keep=False).sum()
print(f"Znaleziono {duplicates} duplikatów w tekst")
df = df[~df['tekst'].duplicated(keep=False)]
print(f"Liczba wierszy po usunięciu: {len(df)}")

# wyrównanie liczby przykładów do najmniejszej wartości
counts = df['intencja'].value_counts()
min_count = counts.min()
df = df.groupby('intencja').apply(lambda x: x.sample(n=min_count)).reset_index(drop=True)

# wczytanie danych lokalnie żeby nie musieć znowu generować
#  df = pd.read_csv('dane2.csv')



# tokenizacja używając tensorflow, uczy sie od zera z moich danych

text = df['tekst']
label = df['intencja']

# nazwy na liczby calkowite
encoder = LabelEncoder()
label_encoded = encoder.fit_transform(label)

# tokenizacja
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
# padding
x = pad_sequences(sequences, maxlen=50)
# one hot encoding
y = to_categorical(label_encoded)

# train test val split
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, stratify=label_encoded, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=2/3, stratify=y_temp, random_state=42
)

# konwersja danych do tensorów PyTorch
x_train_tensor = torch.LongTensor(x_train)
x_val_tensor   = torch.LongTensor(x_val)
x_test_tensor  = torch.LongTensor(x_test)

y_train_tensor = torch.FloatTensor(y_train)
y_val_tensor   = torch.FloatTensor(y_val)
y_test_tensor  = torch.FloatTensor(y_test)

# data loaders do dalszego trenowania w batchach
batch_size = 64

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset  = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# w modelowaniu decyduję się na zastosowanie sieci neuronowych
# oraz LLM ponieważ są to najbardziej optymalne rozwiązania do klasyfikacji tekstu


# model lstm z jednym layerem lstm

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # warstwa embeddingów
        # jednowarstwowy lstm
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        # dropout i warstwy fully connected
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        # zamiana indeksów na embeddingi
        x = self.embedding(x)
        # przepuszczenie przez lstm
        out, _ = self.lstm(x)
        # ostatni ukryty stan
        out = out[:, -1, :]  # take last hidden state
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.output(out)
        return out


# setup modelu
# wybór gpu jeśli się da
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
max_words = 10000
embedding_dim = 100
hidden_dim = 128
max_len = 50
num_classes = y.shape[1]

# utworzenie modelu i funkcji treningowych
model = LSTMModel(max_words, embedding_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training loop z walidają

epochs = 10
for epoch in range(epochs):
    # trenowanie modelu
    model.train()
    train_loss = 0.0 # suma bledow w epoce
    correct = 0 # poprawne predykcje
    total = 0 # wszystkie probki 
    all_preds = [] # wszystkie do f1
    all_labels = [] # poprawne predykcje do f1

    # iteracja po batchach treningowych
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad() # zerowanie gradientów z poprzedniego kroku
        outputs = model(x_batch)  # przejście przez model
        loss = criterion(outputs, torch.max(y_batch, 1)[1]) # funkcja straty
        loss.backward() # propagacja wsteczna
        optimizer.step() # aktualizacja wag

        train_loss += loss.item() # zapis wartosci bledu z batcha 
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0) # zliczanie poprawnych predykcji
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy()) # do f1 score
        all_labels.extend(labels.cpu().numpy())

    # metryki do treningu
    train_acc = 100 * correct / total
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    # walidacja
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels_list = []

    with torch.no_grad(): # bez gradientów
        # iteracja po batchach
        for x_batch, y_batch in val_loader: 
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch) # uzyskanie predykcji
            # obliczenie funkcji straty
            loss = criterion(outputs, torch.max(y_batch, 1)[1])
            val_loss += loss.item()

            # predykcja i porownanie z prawdziwymi etykietami
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(y_batch, 1)
            val_total += labels.size(0) # zliczanie poprawnych kwalifikacji
            val_correct += (predicted == labels).sum().item()
            # zapis predykcji i etykiet do f1 score
            val_preds.extend(predicted.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    # metryki walidacyjne
    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(val_labels_list, val_preds, average='weighted')

    print(f" Epoch {epoch+1}/{epochs}")
    print(f" Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f}")
    print(f" Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")


# finalna ewaluacja modelu
model.eval()
correct = 0 # suma bledow w walidacji
total = 0 # liczba poprawnych predykcji
all_preds = [] 
all_labels = []

with torch.no_grad():
    # iteracja po batchach
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1) # uzyskanie klasy z najwyzszym prawdopodobienstwem
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\n Test Accuracy: {test_acc:.2f}% - F1 Score: {test_f1:.4f}")

# confusion matrix do porównania
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = categories)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title(" Macierz pomyłek")
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa etykieta")

# Formatowanie procentów w komórkach
for text in disp.text_.ravel():
    text.set_text(f"{float(text.get_text()):.1f}")

plt.tight_layout()
plt.show()




# teraz zamiast własnoręcznie tokenizować używamy gotowego polskiego modelu fasttext


lang = 'pl'  
fasttext.util.download_model(lang, if_exists='ignore') # pobieranie modelu
ft = fasttext.load_model(f'cc.{lang}.300.bin') # załadowanie modelu

embedding_dim = 300
print(f"✅ Loaded FastText model for language: {lang}")

# przygotowanie danych
text = df['tekst']
label = df['intencja']

encoder = LabelEncoder() # encoder, zmiana etykiet na liczby
label_encoded = encoder.fit_transform(label)

# keras tokenizer do konwersji na sekwencje
MAX_WORDS = 10000 # maksymalna liczba słów
MAX_LEN = 50 # maksymalna długość sekwencji

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>") # tokenizer dla nieznanych słów
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
x = pad_sequences(sequences, maxlen=MAX_LEN) # padding (ujednolicenie długości)
y = to_categorical(label_encoded) # one hot encoding

word_index = tokenizer.word_index
vocab_size = min(MAX_WORDS, len(word_index) + 1)
print(f"Vocabulary size: {vocab_size}")

# budowa macierzy embeddinng z fasttext
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# dla każdego słowa bierzemy wektor fast text
for word, i in word_index.items():
    if i >= vocab_size:
        continue
    try:
        embedding_matrix[i] = ft.get_word_vector(word)
    except:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

# zmiana macierzy na tensor pytorch
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
print(f"Created embedding matrix of shape: {embedding_matrix.shape}")

# train test val split
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, stratify=label_encoded, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=2/3, stratify=y_temp, random_state=42
)

# zmiana na tensory pytorch
x_train_tensor = torch.LongTensor(x_train)
x_val_tensor   = torch.LongTensor(x_val)
x_test_tensor  = torch.LongTensor(x_test)

y_train_tensor = torch.FloatTensor(y_train)
y_val_tensor   = torch.FloatTensor(y_val)
y_test_tensor  = torch.FloatTensor(y_test)

# loadery danych
batch_size = 64
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset  = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

CLASS_NAMES = list(encoder.classes_)





# taki sam model lstm ale na danych stokenizowanych z fasttext


class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_classes, num_layers=1, freeze=False):
        super(LSTMModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        # fasttext embeddingi (wczytane z gotowej macierzy)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze  # jeśli true, embeddingi nie będą aktualizowane
        )

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.output(out)
        return out


# setup modelu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 128
num_classes = y.shape[1]

# model z FastText embeddingami
model = LSTMModel(embedding_matrix, hidden_dim, num_classes, freeze=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training loop z walidają

epochs = 10
for epoch in range(epochs):
    # trenowanie modelu
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, torch.max(y_batch, 1)[1])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc = 100 * correct / total
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    # walidacja
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels_list = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, torch.max(y_batch, 1)[1])
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(y_batch, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(val_labels_list, val_preds, average='weighted')

    print(f" Epoch {epoch+1}/{epochs}")
    print(f" Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f}")
    print(f" Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")


# finalna ewaluacja modelu
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\n Test Accuracy: {test_acc:.2f}% - F1 Score: {test_f1:.4f}")

# confusion matrix do porównania
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = categories)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title(" Macierz pomyłek")
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa etykieta")

# formatowanie procentów w komórkach
for text in disp.text_.ravel():
    text.set_text(f"{float(text.get_text()):.1f}")

plt.tight_layout()
plt.show()









# bidirectional model z 2 layerami lstm, zmienionym dropoutem i poolingiem
# ten model jest bardziej zaawansowany

class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_classes, num_layers=2, freeze=False):
        super(LSTMModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        # pretrained fasttext embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze
        )

        # 2 warstwowy dwukierunkowy lstm
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, # dwukierunkowy
            dropout=0.2 # zmiana dropoutu
        )

        # głowa klasyfikująca (fully connected layer)
        self.dropout = nn.Dropout(0.2) # zmiana dropoutu
        self.fc1 = nn.Linear(hidden_dim * 2, 128) # *2 bo LSTM jest dwukierunkowy
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        # emebedding (zmiana tokenow na wektory)
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1) # zmieniony pooling (mean pooling)
        # dropout i warstwy w pełni połączone
        out = self.dropout(out) 
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.output(out)
        return out


# setup modelu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 128 # liczba jednostek ukrytych w LSTM
num_classes = y.shape[1] # liczba klas

# utworzenie modelu z embeddingami FastText
model = LSTMModel(embedding_matrix, hidden_dim, num_classes, freeze=False).to(device)

# Funkcja straty i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training loop z walidacją
epochs = 10
for epoch in range(epochs):
    # train
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad() # wyzerowanie gradientów
        outputs = model(x_batch) # predykcja 
        loss = criterion(outputs, torch.max(y_batch, 1)[1]) # obliczenie straty
        loss.backward() # propagacja wsteczna
        optimizer.step() # aktualizacja wag

        # metryki treningowe
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # accuracy i f1
    train_acc = 100 * correct / total
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    # walidacja
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels_list = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, torch.max(y_batch, 1)[1])
            val_loss += loss.item()

            # metryki walidacyjne
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(y_batch, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(val_labels_list, val_preds, average='weighted')

    # wyniki
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f}")
    print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")


# finalna ewaluacja 

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# accuracy i f1
test_acc = 100 * correct / total
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\n Test Accuracy: {test_acc:.2f}% - F1 Score: {test_f1:.4f}")


# confusion matrix do porównania
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = categories)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title(" Macierz pomyłek")
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa etykieta")

# formatowanie procentów w komórkach
for text in disp.text_.ravel():
    text.set_text(f"{float(text.get_text()):.1f}")

plt.tight_layout()
plt.show()







# model fasttext

# Podział na train i temp
X_train_txt, X_temp_txt, y_train_txt, y_temp_txt = train_test_split(
    df['tekst'], df['intencja'], 
    test_size=0.3, 
    stratify=df['intencja'], 
    random_state=42
)

# Podział temp na val i test

X_val_txt, X_test_txt, y_val_txt, y_test_txt = train_test_split(
    X_temp_txt, y_temp_txt, 
    test_size=2/3, 
    stratify=y_temp_txt, 
    random_state=42
)

print(f"Liczebności zbiorów (identyczne jak w LSTM):")
print(f"Train: {len(X_train_txt)}")
print(f"Val:   {len(X_val_txt)}")
print(f"Test:  {len(X_test_txt)}")

# Przygotowanie plików dla fasttext
# FastText uczy się z plików .txt, a nie ze zmiennych pythonowych

def save_to_fasttext_format(filename, texts, labels):
    with open(filename, 'w', encoding='utf-8') as f:
        for text, label in zip(texts, labels):
            # Format: __label__kategoria treść
            clean_label = label.replace(" ", "_")
            clean_text = text.replace("\n", " ").strip()
            f.write(f"__label__{clean_label} {clean_text}\n")

train_file = "ft_train.txt"
val_file = "ft_val.txt" # FastText może używać walidacji do autotune (opcjonalnie)
test_file = "ft_test.txt" # Tego użyjemy do pętli ewaluacyjnej

save_to_fasttext_format(train_file, X_train_txt, y_train_txt)
save_to_fasttext_format(val_file, X_val_txt, y_val_txt)
save_to_fasttext_format(test_file, X_test_txt, y_test_txt)

# Trening modelu
# FastText nie ma pętli "for epoch in range" w Pythonie on robi to w C++ wewnętrznie
# Podajemy mu plik treningowy i liczbę epok

model_ft = fasttext.train_supervised(
    input=train_file, 
    epoch=25, 
    lr=1.0, 
    wordNgrams=2, 
    verbose=0
)

# Finalna ewaluacja
correct = 0
total = 0
all_preds = []
all_labels = []


# Iterujemy po zbiorze testowym przygotowanym w poprzednim kroku (X_test_txt, y_test_txt)
for text, true_label in zip(X_test_txt, y_test_txt):
    # FastText wymaga tekstu bez nowych linii
    clean_text = text.replace("\n", " ")
    
    # Predykcja
    prediction = model_ft.predict(clean_text)
    
    # Wyciągnięcie etykiety i czyszczenie (np. __label__zwrot -> zwrot)
    pred_label_raw = prediction[0][0] 
    pred_label_clean = pred_label_raw.replace("__label__", "").replace("_", " ")
    
    # Zbieranie wyników
    all_preds.append(pred_label_clean)
    all_labels.append(true_label)
    
    if pred_label_clean == true_label:
        correct += 1
    total += 1

# Wyliczenie metryk
test_acc = 100 * correct / total
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\n Native FastText Test Accuracy: {test_acc:.2f}% - F1 Score: {test_f1:.4f}")

# TWykres
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

cm = confusion_matrix(y_true, y_pred, normalize='true', labels=categories) * 100
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title(" Macierz pomyłek - FastText")
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa etykieta")

# Formatowanie procentów w komórkach
for text in disp.text_.ravel():
    text.set_text(f"{float(text.get_text()):.1f}")

plt.tight_layout()
plt.show()

# Sprzątanie plików tymczasowych
if os.path.exists("ft_train.txt"): os.remove("ft_train.txt")
if os.path.exists("ft_val.txt"): os.remove("ft_val.txt")
if os.path.exists("ft_test.txt"): os.remove("ft_test.txt")