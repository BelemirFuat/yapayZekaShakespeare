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

## Zaman Ölçümleri

Eğitim sürecinde her iterasyonun süresi ölçülür ve `time.txt` dosyasına kaydedilir. Bu, modelin eğitim süresi hakkında bilgi verir ve eğitim sürecinin verimliliğini takip etmemize yardımcı olur.

## Sonuçlar

Modelin eğitimi sırasında her iterasyonda kayıp değeri hesaplanır. Aşağıda, modelin eğitim sürecinden alınan örnek çıktı verilmiştir:
step 0: train loss 4.2435, val loss 4.3213 step 1: train loss 4.1221, val loss 4.2876 step 2: train loss 4.0044, val loss 4.2439 ...

Eğitim tamamlandıktan sonra, model tarafından üretilen metinler aşağıdaki gibi olabilir:

step 0: train loss 4.2435, val loss 4.3213 step 1: train loss 4.1221, val loss 4.2876 ...

