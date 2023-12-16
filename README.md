# What is NLP ?
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language.
> "NLP is basically math and statistics with elements of linguistics."
> 
> Shivali Goel
# 🤔 Text Summarization with NLP?
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

summary = " ".join([cümle.text for cümle in doc.sents][:2])  # İlk iki cümleyi alarak özetleme

print(summary)
```
> \[!IMPORTANT]
>
>  Assume that we separete text for extract sentences. 

Code Example for Abstractive summarization with Huggingface

```python
# install Transformers: pip install transformers
from transformers import pipeline
# This dataset is powered by Facebook and has been created to utilize the large dataset of CNN and Daily Mail.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ """
print(summarizer(text, max_length=130, min_length=30, do_sample=False))
```
[Code: medium christianbernecker][1]

We have seen that transformers are used for abstractive analysis.

> \[!NOTE]
>
> A transformer is a deep learning architecture .

[1]: https://christianbernecker.medium.com/nlp-summarization-use-bert-and-bart-to-summarize-your-favourite-newspaper-articles-fb9a81bed016

### 🤯 let's deep dive into transformers?
The original Transformer is based on an encoder-decoder architecture and is a classic sequence-to-sequence model.
There are many ways to solve a given task, some models may implement certain techniques or even approach the task from a new angle, but for Transformer models, the general idea is the same. Owing to its flexible architecture, most models are a variant of an encoder, decoder, or encoder-decoder structure. 
BERT was first designed for machine translation but has become a common model for various language tasks. In text classification, BERT is an encoder-only model, which means it focuses on understanding words from both directions. To use BERT for text classification, a sequence classification head is added. It's a linear layer converting final hidden states to logits for predicting labels. Cross-entropy loss is then calculated to find the most likely label.


**Attention Mechanism:** The fundamental feature of the Transformer is an attention mechanism that enables each element in the input sequence to focus on other elements. This allows each output element, especially words or tokens, to assign weights to all other elements in the input. |

**Encoder-Decoder Architecture**: The Transformer typically consists of two main parts: an encoder and a decoder. The encoder encodes input data in a way that can be better represented. The decoder then transforms this representation into output data.

**Multi-Head Attention:** The attention mechanism is expanded using multiple heads. Each head can focus on different features, allowing the model to learn more complex relationships.

**Layer Normalization and Feedforward Networks:** Each encoder and decoder layer includes layer normalization and a feedforward network. Layer normalization stabilizes training by normalizing the inputs to each layer. The feedforward network helps transform the representations of tokens at each position in a more complex way.
# 💱 Natural Language Processing for Finance
NLP has many applications and benefits in finance, such as the ability to analyze
sentiment and extract information from financial text data, to automate the generation of trading
signals and risk alerts, and to inform decision-making and risk management in the finance
industry.
## Some examples of how NLP is used in finance:

| Usage  | Description                           |
| :---------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| Recognizing Financial Entities | NLP helps us identify and classify named entities in text, such as **people, locations, dates, numbers, etc**. to make recommendations or predictions.|
| Classifying Financial Documents  |  It is capable of automated classification across various **agreement types**, including loans, service agreements, and consulting agreements. It excels in categorizing news based on **ESG** (Environmental, Social, and Governance) criteria. Additionally, it efficiently identifies topics within **banking-related texts** and classifies tickets based on their topic or class. Furthermore, our system can accurately determine **sentiment**, distinguishing between negative, positive, and neutral expressions. |
| Understanding Entities in Context |     Understanding entities in context is the ability of asserting if an entity is mentioned to happen in the present, past, future, if it’s negated, present, absent, if it’s hypothetical, probable, etc. |
| Extracting Financial Relationships |     Relation Extraction is the ability to infer if two entities are connected. It helps us extract relations between a company and its profit, losses, cash flow operations, etc. |
| Normalization and Data Augmentation |  **Normalization** makes words more consistent and helps the model work better. When we normalize text, we make it less random and align it with a set standard.**Data Augmentation** refers to the capability of using extracted information, such as Company Names, to inquire from data sources and acquire additional information. This can include details like the Company's SIC code, Trading Symbol, Address, Financial Period, and more. |

<div style="text-align: center;">
  <p>Firstly, let's look at the nlp investments made in the financial sector:</p>
  <img src="https://github.com/mehmetismostlyclear/NlpFinansRaporu/assets/87391205/7c7483d3-de7d-47cc-a802-4c6fa12f9b8b" alt="Resim" width="500" />
</div>

[Picture: towards data science][2]

[2]: www.towardsdatascience.com/nlp-startup-funding-in-2022 

<div style="text-align: center;">
  <img src="https://github.com/mehmetismostlyclear/NlpFinansRaporu/assets/87391205/0fedcfd2-6145-479e-b6a7-5606ef2ec458" alt="Resim" width="500" />
</div>

[Picture: jhon sonw lab][3]

[3]: https://www.johnsnowlabs.com/examining-the-impact-of-nlp-in-financial-services/
As seen in the graph, NLP is widely employed in the BFSI sector, particularly for automating customer interactions, detecting fraud, and analyzing sentiment. Its extensive use highlights its pivotal role in improving operational efficiency, customer satisfaction, and data-driven decision-making in the finance industry.

For financial institutions, which can be reluctant to deploy cutting-edge techniques like machine learning, this socialization process is an important step. “As more and more people see it work and understand the lingo, they see that it’s not a dark art — it’s math."
