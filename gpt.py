import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import torch.cuda.amp as amp

# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 50000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ------------
print("Cihaz: " + device)  # Cihaz bilgisi yazdırılır (CPU veya GPU)
torch.manual_seed(1337)  # Rastgelelikten sonuçların tekrarlanabilmesi için sabit bir seed belirleniyor

# Metin verisini oku (Tiny Shakespeare)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Metinde bulunan tüm benzersiz karakterleri alıyoruz
chars = sorted(list(set(text)))
vocab_size = len(chars)  # Kelime dağarcığı boyutunu öğreniyoruz
# Karakterden tamsayıya, tamsayıdan karaktere dönüşüm için iki sözlük oluşturuyoruz
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # encoder: bir string alır, tamsayı listesi döndürür
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: tamsayı listesi alır, string döndürür

# Eğitim ve test verisi ayırma
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # Verinin %90'ı eğitim, %10'u doğrulama olarak ayrılıyor
train_data = data[:n]
val_data = data[n:]

# Veri yükleme fonksiyonu
def get_batch(split):
    # Küçük bir batch oluşturuyoruz, girişler x ve hedefler y olacak
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # rastgele bir başlangıç noktası seçiyoruz
    x = torch.stack([data[i:i+block_size] for i in ix])  # giriş verilerini oluşturuyoruz
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # hedef verilerini oluşturuyoruz
    x, y = x.to(device), y.to(device)  # verileri doğru cihaza taşıyoruz
    return x, y

# Modelin kaybını tahmin etme (eğitim sırasında doğrulama için)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # Modeli değerlendirme moduna alıyoruz
    for split in ['train', 'val']:  # hem eğitim hem de doğrulama verisi üzerinde çalışacağız
        losses = torch.zeros(eval_iters)  # her bir değerlendirme için kayıpları tutacak bir vektör
        for k in range(eval_iters):
            X, Y = get_batch(split)  # Bir batch verisi alıyoruz
            logits, loss = model(X, Y)  # Modelden tahminler ve kayıp değerini alıyoruz
            losses[k] = loss.item()  # Kayıpları kaydediyoruz
        out[split] = losses.mean()  # Ortalama kaybı hesaplıyoruz
    model.train()  # Modeli tekrar eğitim moduna alıyoruz
    return out

class Head(nn.Module):
    """Self-attention başlıklarından biri"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # Anahtarlar için bir linear katman
        self.query = nn.Linear(n_embd, head_size, bias=False)  # Sorgular için bir linear katman
        self.value = nn.Linear(n_embd, head_size, bias=False)  # Değerler için bir linear katman
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # Düşey maskeyi kaydediyoruz

        self.dropout = nn.Dropout(dropout)  # Dropout katmanı

    def forward(self, x):
        B, T, C = x.shape  # Batching boyutlarını alıyoruz
        k = self.key(x)   # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # Attention skorlarını hesaplıyoruz (affinity hesaplama)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Gelecek adımlara bakmamamız için maskeleme yapıyoruz
        wei = F.softmax(wei, dim=-1)  # Softmax ile olasılıkları hesaplıyoruz
        wei = self.dropout(wei)  # Dropout uyguluyoruz
        # Değerler ile ağırlıklı toplamı alıyoruz
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """Paralel olarak birden fazla self-attention başlığı"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # Birden fazla başlık
        self.proj = nn.Linear(head_size * num_heads, n_embd)  # Çıkışı tek bir boyuta indiriyoruz
        self.dropout = nn.Dropout(dropout)  # Dropout katmanı

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Başlıkları birleştiriyoruz
        out = self.dropout(self.proj(out))  # Son projeksiyon işlemi
        return out

class FeedFoward(nn.Module):
    """Basit bir lineer katman ve sonra bir aktivasyon fonksiyonu"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # ReLU aktivasyonu
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),  # Dropout
        )

    def forward(self, x):
        return self.net(x)  # İleri yönde geçiş

class Block(nn.Module):
    """Transformer bloğu: iletişim ve hesaplama işlemleri"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # Çok başlıklı attention katmanı
        self.ffwd = FeedFoward(n_embd)  # Feedforward katmanı
        self.ln1 = nn.LayerNorm(n_embd)  # İlk normalizasyon
        self.ln2 = nn.LayerNorm(n_embd)  # İkinci normalizasyon

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Self-attention ve layer norm işlemi
        x = x + self.ffwd(self.ln2(x))  # Feedforward ve layer norm işlemi
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Tokenlar için embedding tablosu
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # Transformer katmanları
        self.ln_f = nn.LayerNorm(n_embd)  # Son normalizasyon katmanı
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Son çıkış katmanı

        # Ağırlıkları başlatma (daha iyi sonuçlar için)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Girdi ve hedefler (B, T) boyutunda olacak
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # Transformer katmanları
        x = self.ln_f(x)  # Son normalizasyon
        logits = self.lm_head(x)  # Çıkışlar (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)  # Kayıp fonksiyonu

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Yeni token üretme fonksiyonu
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Son 'block_size' kadar veriyi kullan
            logits, loss = self(idx_cond)  # Tahminleri al
            logits = logits[:, -1, :]  # Sadece son zaman adımını al
            probs = F.softmax(logits, dim=-1)  # Softmax ile olasılıkları al
            idx_next = torch.multinomial(probs, num_samples=1)  # Olasılıklara göre bir token seç
            idx = torch.cat((idx, idx_next), dim=1)  # Yeni token'ı ekleyip devam et
        return idx

# Modeli başlatıyoruz
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parametreler')

# Optimizer oluşturuyoruz
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
file = open("time.txt", "w")
file2 = open("stepTrainLossValLoss.txt", "w")
for iter in range(max_iters):
    start_time = time.time()
    
     # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        file2.write(f"{iter} {losses['train']:.4f} {losses['val']:.4f}\n")

    # Eğitim için bir batch verisi alıyoruz
    xb, yb = get_batch('train')

    # Modeli eğitiyoruz
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    end_time = time.time()
    print(f"{iter}. Time taken for iteration: {end_time - start_time:.2f} seconds")
    file.write(f"{end_time - start_time:.2f}\n")
file.close()

# Modelden metin üretme
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
file2.close()
file3 = open("outputGPT.txt", "w")
file3.write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
file3.close()
