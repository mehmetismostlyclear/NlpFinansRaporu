# NLP Nedir?

Doğal Dil İşleme (NLP), bilgisayarların insan dilini anlaması, yorumlaması, üretmesi ve cevap vermesi amacıyla kullanılan bir yapay zeka dalıdır. NLP, metin madenciliği, dil modelleri ve konuşma tanıma gibi teknikleri içerir. Bu teknoloji, metin tabanlı verileri anlama ve işleme yeteneği sayesinde geniş bir uygulama alanına sahiptir.

# NLP ile Text Summarization Nasıl Yapılır?

NLP'nin metin özetleme yetenekleri, uzun metinleri kısaltma ve önemli bilgileri çıkarma amacıyla kullanılır.
> "NLP is basically math and statistics with elements of linguistics."
> Shivali Goel
> This is the second line.
>
> And this is the third line.
```python
import spacy

# spaCy'nin İngilizce dil modelini yükleyin
nlp = spacy.load("en_core_web_sm")

# Özetlenecek metni tanımlayın
metin = """
Buraya özetlenecek uzun bir metin ekleyin.
"""

# Metni işleyin
doc = nlp(metin)

# Anahtar cümleleri ve bilgileri özetleme
ozet = " ".join([cümle.text for cümle in doc.sents][:2])  # İlk iki cümleyi alarak özetleme

# Sonucu yazdırın
print(ozet)
