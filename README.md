# Data Science Project 41 — Sahte Haber Tespiti (Bag of Words + Naive Bayes)

**Modül**: ML-04 (Sınıflandırma 2) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Bir medya platformunda **data scientist** olarak çalışıyorsun. Platforma her gün binlerce haber makalesi düşüyor ve bunların bir kısmı **sahte/yanıltıcı (fake news)**: uydurma olaylar, komplo teorileri, sansasyonel başlıklar ("ŞOK! Sana söylemedikleri gerçek..."). Bu içerikler hızla yayılıyor ve platformun güvenilirliğini zedeliyor. Şirket, yayınlanan haberleri **FAKE** ve **REAL** olarak ayıran bir model kurmanı istiyor.

Senin görevin: bir haber metnine bakıp **sahte mi, gerçek mi** olduğunu tahmin eden bir model kurmak. Doğru çalışırsa şüpheli içerikler editör onayına düşer, gerçek haberler engellenmeden yayınlanır.

Bunun için **Bag of Words (kelime torbası)** yaklaşımını kullanacaksın — metin sınıflandırmanın klasik ve şaşırtıcı derecede güçlü baseline'ı. Fikir basit: her belgedeki kelimeleri **sayarsın** (sırayı umursamadan, sadece "kaç kez geçti"). Sonra **MultinomialNB** (kelime sayımları için Naive Bayes) bu sayımlardan FAKE/REAL kararı verir. Sahte haberler belli kelime kalıplarını ("shocking", "they", "hillary", "establishment" vb.) çok daha sık kullandığı için bu basit yöntem oldukça başarılıdır.

Veride makalelerin **~%50'si fake**, geri kalanı gerçek — yani **dengeli (balanced)** bir problem. Bu yüzden accuracy de anlamlı; ama yine de precision/recall/f1'e bakacağız.

Bu projede ML-04 dersinde öğrendiklerini birleştirip uygulayacaksın:
- ✅ **Gerçek dünya veri çekme** (URL → CSV cache)
- ✅ **Metin → sayı** dönüşümü (`CountVectorizer` — **bag of words**, kelime SAYIMI)
- ✅ **MultinomialNB** (metin için Naive Bayes)
- ✅ **stop_words + max_features** (sözlüğü temizleme ve sınırlama)
- ✅ **Stratified train/test split**
- ✅ **Precision / Recall / F1**
- ✅ **`feature_log_prob_` ile "en fake / en real kelimeler"**
- ✅ **Bag of Words vs TF-IDF** karşılaştırması
- ✅ **sklearn Pipeline** (vektörize + model tek akışta)

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-41

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 721 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📚 Dataset — Fake or Real News

### Kaynak
- **Mirror / dağıtım**: [lutzhamel/fake-news](https://github.com/lutzhamel/fake-news) GitHub deposu
- **Orijinal derleme**: George McIntire'ın `fake_or_real_news` veri seti (DataCamp "fake news classification" çalışmasından yaygınlaşmıştır)

### Veri Çekme Yöntemi (önemli)
Veri seti **repo'da YOK** — projeyi clone ettiğin gün **runtime'da indireceksin**. Gerçek dünyada veriler S3 / API / CDN üzerinden gelir; statik dosya olarak gönderilmez. Bu yüzden `fetch_news_data()` fonksiyonun:

1. Hedef URL: `https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv`
2. `data/` klasörüne CSV'yi indirir (`requests.get` — dosya ~30MB olduğu için `timeout=120`)
3. Cache: dosya zaten varsa **tekrar indirme**, hemen path döndür

> **Not — ilk indirme:** Dosya ~30MB. İlk test internet ister ve birkaç saniye sürebilir. Sonraki tüm testler cache'den okur.

### Boyut & Target
- **~6335 makale × (id + 3 sütun)**
- Sütunlar: isimsiz **id** sütunu (atılır), `title`, `text`, `label`
- Target: `label` (`REAL` / `FAKE`) — haber gerçek mi sahte mi?
- Class dağılımı:
  - `REAL`: ~3171 (%50) ← gerçek haber
  - `FAKE`: ~3164 (%50) ← sahte haber
- **Dengeli (balanced)** — accuracy anlamlı, yine de precision/recall/f1'e bak.

### Dosya Formatı
- **CSV** (virgülle ayrılmış), başlık satırı VAR
- İlk sütun **isimsiz bir id** (`Unnamed: 0`) — modellemede kullanılmaz, **at**.
- `pd.read_csv(path)` ile oku, ilk sütunu düşür, `['title','text','label']` bırak.

### Örnek Satırlar (kavramsal)
```
label | title                              | text
REAL  | Senate passes budget bill          | The Senate voted on Tuesday to ...
FAKE  | SHOCKING truth they hide from you  | Sources reveal the secret elite ...
```
FAKE makalelerde **SHOCKING**, **TRUTH**, büyük harf vurgusu, komplo tonu ve "share before they delete this" gibi harekete çağıran ifadeler sık görülür. Modelin tam da bu kalıpları öğrenmesini bekliyoruz.

### Domain Notu
Veri seti ABD siyaseti ağırlıklı 2016 dönemi haberlerinden derlenmiştir. FAKE örnekler tipik dezenformasyon kalıplarını (sansasyonel başlık, kaynaksız iddia, duygusal dil), REAL örnekler ise ölçülü gazetecilik dilini yansıtır. Bu yüzden **kelime sayımı (bag of words)** tek başına çok güçlü bir sinyaldir.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **14 fonksiyonu** sırayla doldur. Her task altta testler pass olana kadar düzenlenmeli.

1. **`fetch_news_data(cache_dir='data')`** — URL'den CSV indir, path döndür (cache'li)
2. **`load_news_data(path)`** — `pd.read_csv`, id sütununu at, `['title','text','label']` döndür
3. **`explore_data(df)`** — total, real_count, fake_count, fake_rate
4. **`encode_labels(df)`** — `label`: FAKE→1, REAL→0 (yeni `target` sütunu)
5. **`split_data(X, y)`** — 80/20, stratify=y, random_state=42
6. **`build_bow_pipeline()`** — CountVectorizer(stop_words='english', max_features=10000) + MultinomialNB ← **bag of words, TF-IDF değil**
7. **`train_model(pipe, X_train, y_train)`** — fit + dön
8. **`evaluate_model(pipe, X_test, y_test)`** — accuracy, precision, recall, f1, confusion matrix
9. **`predict_article(pipe, text)`** — tek haber için tahmin + fake olasılığı
10. **`top_fake_words(pipe, n=10)`** — `feature_log_prob_` farkıyla en fake n kelime
11. **`top_real_words(pipe, n=10)`** — tersi, en real n kelime
12. **`compare_bow_vs_tfidf(...)`** — CountVectorizer vs TfidfVectorizer F1 karşılaştır
13. **`predict_batch(pipe, articles)`** — birden çok haber için tahmin listesi
14. **`run_pipeline()`** — uçtan uca akış, özet dict dön

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] Gerçek dünyada **veri çekme** (URL + cache) yapabileceksin
- [x] **Metni sayıya** çevirebileceksin (`CountVectorizer`, **bag of words**)
- [x] **MultinomialNB** ile metin sınıflandırma yapabileceksin
- [x] **stop_words** ve **max_features** ile sözlüğü temizleyip sınırlayabileceksin
- [x] **Stratified split** ile sınıf oranını koruyabileceksin
- [x] **Precision / Recall / F1** hesaplayabileceksin
- [x] **`feature_log_prob_`** ile modelin "en fake / en real kelimelerini" çıkarabileceksin
- [x] **Bag of Words vs TF-IDF** farkını ölçebileceksin
- [x] **sklearn Pipeline** ile vektörize + model adımlarını tek akışta birleştirebileceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (16 test)

Tümü pass olmalı:
- Dataset fetch (internet gerekli, ilk test ~30MB indirir)
- Satır sayısı (~6335) ve sütunlar (`title`, `text`, `label`) doğru mu
- Label değerleri `{'REAL','FAKE'}` mı, fake oranı ~%50 mi
- `target` 0/1 encode edilmiş mi (FAKE=1, REAL=0)
- Stratified split korunmuş mu
- Pipeline tipi doğru mu (**CountVectorizer** + MultinomialNB, `stop_words='english'`)
- F1 > 0.85 (bag of words güçlü baseline)
- Fake-tonlu / real-tonlu örnek haberler doğru yönde sınıflanıyor mu
- En fake / en real kelimeler farklı mı

## 📊 Beklenen Sonuçlar

```
Fake rate: ~%50 (dengeli)
Test F1: ~0.88-0.93 (Bag of Words + MultinomialNB güçlü baseline)
BoW vs TF-IDF: ikisi de ~0.88+ — BoW güçlü bir baseline
En fake kelimeler: sansasyonel / komplo / siyasi vurgu kelimeleri
En real kelimeler: ölçülü gazetecilik dili
```

## 💡 İpuçları

- **İlk testte internet** gerekli (~30MB CSV indirme). Sonraki testler cache'den okur.
- `pd.read_csv(path)` sonra ilk (isimsiz) sütunu at: `df.drop(columns=[df.columns[0]])` veya `index_col=0`
- Pipeline ham string listesi alır; CountVectorizer otomatik vektörize eder — sen elle vektörize etme
- `predict_article`'ta tek string değil **liste** geç: `pipe.predict_proba([text])`
- **Bag of Words** = `CountVectorizer` (kelime SAYIMI). TF-IDF kullanma (o ayrı bir task'ta karşılaştırılıyor).
- `feature_log_prob_[1]` = FAKE, `[0]` = REAL. Farkları en büyük kelimeler en "fake"
- `np.argsort(diff)[::-1][:n]` → en büyük n değerin indeksleri
- `precision_score`, `recall_score`, `f1_score` — hepsi `sklearn.metrics`

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` değerini değiştirme (testler fail olur)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- `data/` klasörü repo'ya **gitmez** (.gitignore'da exclude)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
