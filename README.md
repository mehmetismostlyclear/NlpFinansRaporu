# What is NLP ?
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language.
> "NLP is basically math and statistics with elements of linguistics."
> 
> Shivali Goel
# Text Summarization with NLP?
The aim is to create a short and clear summary while keeping the main meaning of the original content intact. This helps quickly understand the main points without reading the entire text.
### What is the role of NLP in text summarization?
* Automatically shorts documents, papers, podcasts.

* Enables a more in-depth analysis and understanding of the content of news.

* Can provide advanced analytical insights from the data.
* Artificial intelligence algorithms make objective decisions.
* Integrates and consolidates data in one place, allowing access to a wide range of data from a single location.

There are two main approaches to NLP summarization: **extractive summarization** and **abstractive summarization**.
* **Extractive sumarization** involves identifying the most important sentences in the original text and extracting them to create summary. This approach relies on machine learning techniques to identify the most relevant information in the text.
* **Abstractive summarization** involves generating new sentences that capture the essence of the original text. This approach requires a deep understanding of the language and context of the original text, and often involves the use of advanced NLP techniques such as natural language generation.

Code example for Extractive summarization with BERT

```python
# Encode the sentences using the SBERT model
sentence_embeddings = model.encode(sentences)

# Calculate the cosine similarity between each sentence embedding
similarity_matrix = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            similarity_matrix[i][j] = util.cos_sim(sentence_embeddings[i], sentence_embeddings[j])
print(similarity_matrix)
sentence_scores = np.sum(similarity_matrix, axis=1)

# Select the top 3 sentences with the highest scores
summary_sentences = []
for i in range(3):
    index = np.argmax(sentence_scores)
    summary_sentences.append(sentences[index].strip())
    sentence_scores[index] = -1

# Concatenate the summary sentences to create a summary
summary = '. '.join(summary_sentences)
print(summary)
doc = nlp(metin)

# Anahtar cümleleri ve bilgileri özetleme
ozet = " ".join([cümle.text for cümle in doc.sents][:2])  # İlk iki cümleyi alarak özetleme

# Sonucu yazdırın
print(ozet)
```
1^ Assume that we separete text for extract sentences.
Code Example for Abstractive summarization with Huggingface

kaynak düzenlencek(https://christianbernecker.medium.com/nlp-summarization-use-bert-and-bart-to-summarize-your-favourite-newspaper-articles-fb9a81bed016)
```python
# install Transformers: pip install transformers
from transformers import pipeline
# This dataset is powered by Facebook and has been created to utilize the large dataset of CNN and Daily Mail.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ """
print(summarizer(text, max_length=130, min_length=30, do_sample=False))
```
##Okey but what is transformers?
