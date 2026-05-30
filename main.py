import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import os 

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu' # GPU,MPS,CPU
batch_size = 64
block_size = 128
n_embd = 384 # embeding dimension
n_head = 6 # the number of heads we would like
n_layer = 6 
dropout = 0.2
max_iters = 15000 
eval_interval = 1000
learning_rate = 1e-4 
eval_iters = 100

torch.manual_seed(1337) 

###
languages_by_code = {
    "ar": "arabic",
    "bg": "bulgarian",
    "de": "german",
    "el": "modern greek",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "it": "italian",
    "ja": "japanese",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sw": "swahili",
    "th": "thai",
    "tr": "turkish",
    "ur": "urdu",
    "vi": "vietnamese",
    "zh": "chinese"
}
print("Dostępne języki:")
for code, language in languages_by_code.items():
    print(f"  {code}: {language}")

# Preparing data
text = pd.read_csv('train_data.csv')
unique_languages = sorted((list(set(text['labels'])))) 
unique_languages_id = {lang: i for i, lang in enumerate(unique_languages)}
unique_languages_txt = {i: lang for i, lang in enumerate(unique_languages)}
text['labels'] = text['labels'].map(unique_languages_id) # update language to int, example: pl -> 0
# Tokenization
all_text = "".join(text['text'])
chars = sorted(list(set(all_text))) # unique chars in texts
chars.insert(0, '[PAD]')
vocab_size = len(chars)
# Mapping characters to integers
stoi = {ch:i for i,ch in enumerate(chars)} # encoder 
encode = lambda s: [stoi[c] for c in s] # string to ints - funkcja kodująca 

# Train data 
PAD = 0 # Additional token
batch = []
for rows in text['text']:
    encoded = encode(rows)
    if len(encoded) > block_size:
        encoded = encoded[:block_size]
    else:
        encoded += ([PAD] * (block_size - len(encoded)))
    batch.append(encoded)

X = torch.tensor(batch, dtype=torch.long)
Y = torch.tensor(text['labels'].values, dtype=torch.long)
n = int(0.9 * len(X))
X_train = X[:n]
X_val = X[n:]
Y_train = Y[:n]
Y_val = Y[n:]

# Data loading
def get_batch(split):
    # Appropriate set
    X_data = X_train if split == 'train' else X_val
    Y_data = Y_train if split == 'train' else Y_val

    # Random indez for batch
    ix = torch.randint(0, len(X_data), (batch_size,))

    x = X_data[ix]
    y = Y_data[ix]

    # Send batch to device
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad() # Disable history tracking and do not calculate gradients
def estimate_loss():
    out = {}
    m.eval() # Stop learning
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__() # nn.Module Function
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        head_size = k.shape[-1]
        # compute attention socres
        wei = q @ k.transpose(-2, -1) * (head_size**-0.5)
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 *  n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class LanguageClassifier(nn.Module):

    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, num_classes)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x)    
        x = self.ln_f(x)      
        x = x.mean(dim=1) 

        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

num_classes = len(unique_languages)

model = LanguageClassifier(vocab_size, num_classes)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

MODEL_PATH = 'language_classifier.pth'
if os.path.exists(MODEL_PATH):
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"Krok {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(m.state_dict(), 'language_classifier.pth')

def predict_language(user_sentence):
    m.eval()
    with torch.no_grad():

        encoded = [stoi[c] for c in user_sentence.lower() if c in stoi]
        
        max_len = block_size 
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded += [PAD] * (max_len - len(encoded))
            
        input_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
        
        logits, _ = m(input_tensor)
        probs = F.softmax(logits, dim=-1)
        
        predicted_id = torch.argmax(probs, dim=-1).item()
        
        predicted_lang = unique_languages[predicted_id]
        confidence = probs[0][predicted_id].item() * 100
        
        return predicted_lang, confidence

while True:
    user_input = input("\nWpisz zdanie w dowolnym języku ('q' aby wyjść, lub 't' aby sprawdzić dane testowe): ")
    if user_input.lower() == 'q':
        break
    elif user_input.lower() == 't':
        test = pd.read_csv('test_data.csv')
        count = len(test)
        right = 0 
        for row in test.itertuples():
            lang, conf = predict_language(row.text)
            if row.labels == lang:
                right += 1
        print(f"Skuteczność: {right/count * 100}%")
    else:
        lang, conf = predict_language(user_input)
        print(f"Werdykt modelu: Język to prawdopodobnie '{lang}' (Pewność: {conf:.2f}%)")

