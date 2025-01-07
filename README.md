# Yapay Zeka Projesi
Bu proje, PyTorch kullanarak bir GPT (Generative Pretrained Transformer) dil modeli geliştirmeyi amaçlamaktadır. Model, verilen metin verisinden öğrenerek metin üretme yeteneği kazanmaktadır. Proje, modelin eğitimi, metin üretimi ve eğitim sürecinin zaman ölçümleri gibi çeşitli bileşenleri içerir.

## Proje Amacı ve Yapısı

Bu projenin amacı, bir dil modeli geliştirerek metin verisinden öğrenmek ve verilen bir başlangıç bağlamına göre yeni metinler üretmektir. Projeye temel olarak Transformer mimarisine dayalı GPT modelini eğitmek için `input.txt` adlı Tiny Shakespeare veri seti kullanılmıştır. Model, metin verisini okuyarak bir dil modeli olarak eğitim alır ve ardından verilen bir başlangıç bağlamına dayalı olarak metin üretir.

## Modelin Temel Bileşenleri

### 1. Veri Hazırlığı
- Modelin eğitilmesi için `input.txt` dosyasındaki metin verisi okunur.
- Metindeki benzersiz karakterler sıralanır ve her bir karakter için bir tamsayı karşılıkları (`stoi` ve `itos`) oluşturulur.
  
### 2. Modelin Yapısı
Modelin temel yapı taşları şunlardır:

- **Token Embedding ve Pozisyonel Embedding**: Modelin her token (kelime veya karakter) için embedding (gömülü vektör) katmanları vardır. Ayrıca, her token'ın pozisyonu için pozisyonel embedding katmanı da bulunmaktadır.

- **Self-Attention (Head ve MultiHeadAttention)**: Transformer tabanlı mimaride, modelin farklı başlıklar altında her token’a olan ilişkisini öğrenebilmesi için self-attention mekanizması kullanılır. Bu, modelin hangi token'ların birbirleriyle ilişkili olduğunu anlamasına yardımcı olur.

- **FeedForward Katmanı**: Her bir transformer bloğunda, self-attention işleminin ardından feedforward katmanı kullanılarak daha derin öğrenme yapılır.

- **Block ve Transformer Yapısı**: Model, birden çok `Block`'tan oluşur. Her bir blok, self-attention katmanı ve feedforward katmanı içerir.

- **Son Katman (Output Layer)**: Modelin sonunda, her token için logits hesaplanır ve bu logits üzerinden bir kelime dağılımı elde edilir.

### 3. Eğitim Süreci
Modelin eğitiminde kullanılan adımlar şunlardır:
- **Veri Partileri (Batching)**: Modelin eğitiminde her seferinde küçük veri partileri (batch) kullanılır. Bu partiler `get_batch` fonksiyonu ile oluşturulur.
  
- **Kayıp Hesaplama**: Eğitim sırasında, modelin tahmin ettiği logits ile gerçek hedefler karşılaştırılır ve bu karşılaştırma sonucu kayıp değeri hesaplanır. Bu kayıp değeri, çapraz entropi kaybı (cross-entropy loss) olarak belirlenmiştir.

- **Optimizasyon**: Modelin parametreleri `AdamW` optimizasyon algoritması ile güncellenir.

- **Zaman Ölçümü**: Eğitim sırasında her iterasyonun süresi ölçülüp `time.txt` dosyasına kaydedilir.

### 4. Metin Üretimi
Model, eğitim tamamlandıktan sonra verilen bir başlangıç bağlamı üzerinden yeni metinler üretir. Bu süreç, `generate` fonksiyonu ile gerçekleştirilir. Model, başlangıç bağlamından itibaren yeni token'lar üretir ve metin devam eder.

## Kodun Genel Yapısı

### 1. Veri Hazırlığı
Modelin eğitimine başlamadan önce, metin verisi `input.txt` dosyasından okunur ve her karakter için birer tamsayı karşılığı (`stoi` ve `itos`) oluşturulur. Bu, modelin metni öğrenebilmesi için gereklidir.

### 2. Modelin Tanımlanması
Modelin yapı taşları, aşağıdaki bileşenlerden oluşur:
- **Token Embedding**: Kelimelerin gömülü vektör temsilleri.
- **Pozisyon Embedding**: Kelimelerin sıralarını temsil eden embedding katmanı.
- **Self-Attention Katmanları**: Çok başlıklı self-attention mekanizması.
- **FeedForward Katmanları**: Self-attention sonrası kullanılan derin öğrenme katmanları.
- **Son Katman**: Modelin çıktısını verir.

### 3. Eğitim Döngüsü
Eğitim döngüsünde, model her iterasyonda verileri işler ve kayıp hesaplanarak geri yayılım (backpropagation) yapılır. Bu işlem, modelin parametrelerini güncelleyerek daha iyi tahminler yapmasını sağlar. Eğitim sırasında zaman ölçümleri yapılır ve her 5000 iterasyonda eğitim ve doğrulama kayıpları yazdırılır.

### 4. Metin Üretimi
Eğitim tamamlandıktan sonra, `generate` fonksiyonu kullanılarak modelden yeni metinler üretilir. Başlangıç bağlamı, modelin metin üretme sürecine yardımcı olur ve model verilen bağlamdan yeni token'lar üretir.

## 5. Zaman Ölçümleri

Eğitim sürecinde her iterasyonun süresi ölçülür ve `time.txt` dosyasına kaydedilir. Bu, modelin eğitim süresi hakkında bilgi verir ve eğitim sürecinin verimliliğini takip etmemize yardımcı olur.

## 6. Kodun işlevi

### 6.1 Kütüphaneler

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
```

- torch: PyTorch kütüphanesi, derin öğrenme modelleri oluşturmak ve eğitmek için kullanılır.
- torch.nn: PyTorch'un sinir ağı modülleri.
- torch.nn.functional: PyTorch'un fonksiyonel API'si, çeşitli aktivasyon fonksiyonları ve kayıp fonksiyonları içerir.
- time: Zaman ölçümleri için Python'un standart kütüphanesi.

### 6.2 Hiperparametreler

```python
batch_size = 128
block_size = 128 
max_iters = 5000
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
```

- batch_size: Paralel olarak işlenecek bağımsız dizilerin sayısı.
- block_size: Tahminler için maksimum bağlam uzunluğu.
- max_iters: Maksimum eğitim iterasyonu sayısı.
- eval_interval: Değerlendirme aralığı.
- learning_rate: Öğrenme oranı.
- device: Eğitim için kullanılacak cihaz (CPU veya GPU).
- eval_iters: Değerlendirme iterasyonu sayısı.
- n_embd: Gömme boyutu.
- n_head: Çok başlıklı dikkat mekanizmasındaki başlık sayısı.
- n_layer: Transformer katman sayısı.
- dropout: Dropout oranı.

### 6.3 Rastgelelik ve veri hazırlığı
```python
print("Cihaz: " + device)
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
```
- torch.manual_seed(1337): Rastgelelikten sonuçların tekrarlanabilmesi için sabit bir seed belirleniyor.
- with open('input.txt', 'r', encoding='utf-8') as f: Metin verisini okuma (Tiny Shakespeare).
- chars: Metinde bulunan tüm benzersiz karakterler.
- vocab_size: Kelime dağarcığı boyutu.
- stoi ve itos: Karakterden tamsayıya ve tamsayıdan karaktere dönüşüm için iki sözlük.
- encode ve decode: String ve tamsayı listesi arasında dönüşüm yapan fonksiyonlar.
- data: Metin verisinin tamsayıya dönüştürülmüş hali.
- train_data ve val_data: Eğitim ve doğrulama verisi olarak ayrılmış veri.

### 6.4 Batch oluşturma fonksiyonu

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

- get_batch(split): Eğitim veya doğrulama verisi için bir batch oluşturur.
- data: Eğitim veya doğrulama verisi.
- ix: Rastgele bir başlangıç noktası seçer.
- x ve y: Giriş ve hedef verileri.
- x, y = x.to(device), y.to(device): Verileri doğru cihaza taşır.

### 6.5 Kayıp tahmin fonksiyonu
```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

- estimate_loss(): Eğitim ve doğrulama verisi üzerinde model kaybını tahmin eder.
- model.eval(): Modeli değerlendirme moduna alır.
- losses: Her bir değerlendirme için kayıpları tutar.
- model.train(): Modeli tekrar eğitim moduna alır.

### 6.6 Model sınıfları

#### 6.6.1 Head sınıfı

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
```
- Head: Self-attention başlıklarından biri.
- key, query, value: Anahtar, sorgu ve değerler için linear katmanlar.
- tril: Düşey maskeyi kaydeder.
- forward: İleri yönde geçiş işlemi.

#### 6.6.2 MultiHeadAttention Sınıfı
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

- MultiHeadAttention: Paralel olarak birden fazla self-attention başlığı.
- heads: Birden fazla başlık.
- proj: Çıkışı tek bir boyuta indirir.
- forward: İleri yönde geçiş işlemi.

#### 6.6.3 Feedforward Sınıfı
```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```
- FeedFoward: Basit bir lineer katman ve aktivasyon fonksiyonu.
- net: Lineer katmanlar ve ReLU aktivasyonu.
- forward: İleri yönde geçiş işlemi.

#### 6.6.4 Block Sınıfı
```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```
- Block: Transformer bloğu.
- sa: Çok başlıklı attention katmanı.
- ffwd: Feedforward katmanı.
- ln1, ln2: Layer normalization katmanları.
- forward: İleri yönde geçiş işlemi.

#### GPTLanguageModel Sınıfı
```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
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
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

- GPTLanguageModel: GPT dil modeli.
- token_embedding_table: Tokenlar için embedding tablosu.
- position_embedding_table: Pozisyon embedding tablosu.
- blocks: Transformer katmanları.
- ln_f: Son normalizasyon katmanı.
- lm_head: Son çıkış katmanı.
- _init_weights: Ağırlıkları başlatma fonksiyonu.
- forward: İleri yönde geçiş işlemi.
- generate: Yeni token üretme fonksiyonu.

### 6.7 Modelin eğitimi ve değerlendirmesi
```python
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parametreler')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
file = open("time.txt", "w")
file2 = open("stepTrainLossValLoss.txt", "w")
for iter in range(max_iters):
    start_time = time.time()
    
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        file2.write(f"{iter} {losses['train']:.4f} {losses['val']:.4f}\n")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    end_time = time.time()
    print(f"{iter}. Time taken for iteration: {end_time - start_time:.2f} seconds")
    file.write(f"{end_time - start_time:.2f}\n")
file.close()
```

- model = GPTLanguageModel(): Modeli başlatır.
- m = model.to(device): Modeli doğru cihaza taşır.
- optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate): Optimizer oluşturur.
- Eğitim döngüsü: Modeli eğitir ve her eval_interval adımda kayıpları değerlendirir.

### 6.8 Metin Üretimi
```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
file2.close()
file3 = open("outputGPT.txt", "w")
file3.write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
file3.close()
```
- context = torch.zeros((1, 1), dtype=torch.long, device=device): Başlangıç bağlamı.
- m.generate(context, max_new_tokens=500): Modelden yeni metin üretir.
- Üretilen metni dosyaya yazar.

## Sonuçlar

Modelin eğitimi sırasında her iterasyonda kayıp değeri hesaplanır. Aşağıda, modelin eğitim sürecinden alınan örnek çıktı verilmiştir:
step 0: train loss 4.2435, val loss 4.3213 
step 500: train loss 4.1221, val loss 4.2876 
step 1000: train loss 4.0044, val loss 4.2439 ...

Eğitim tamamlandıktan sonra, model tarafından üretilen metinler aşağıdaki gibi olabilir:

step 0: train loss 4.2435, val loss 4.3213 step 1: train loss 4.1221, val loss 4.2876 ...

