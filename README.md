# What is NLP ?

Natural Language Processing (NLP) is a field of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language.
> "NLP is basically math and statistics with elements of linguistics."
> 
> Shivali Goel
#### Brief overview:
- [x] 💨 **Tokenization**:Breaking down a sentence into individual words or tokens, making it easier for analysis.
- [x] 💬 **Part-of-Speech (POS) Tagging**: Assigning grammatical tags (such as noun, verb, adjective) to each word in a sentence for better understanding.
- [x] 👀 **Named Entity Recognition (NER)**:Identifying and classifying entities like names, locations, and organizations within a text.
- [x] 🈳 **Stemming and Lemmatization**: Simplifying words to their base or root form to reduce variations and aid in analysis.
- [x] 😥 **Sentiment Analysis**: Determining the emotional tone of a piece of text, often categorized as positive, negative, or neutral.
- [x] 🗣️ **Syntax and Grammar Parsing**: Analyzing the grammatical structure of sentences to comprehend their meaning.

# Text Summarization with NLP?
The aim is to create a short and clear summary while keeping the main meaning of the original content intact. This helps quickly understand the main points without reading the entire text.
###  🤔 What is the role of NLP in text summarization?
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
> A transformer is a deep learning architecture model.

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

[Picture: towardsdatascience][2]

[2]: www.towardsdatascience.com/nlp-startup-funding-in-2022 

<div style="text-align: center;">
  <img src="https://github.com/mehmetismostlyclear/NlpFinansRaporu/assets/87391205/0fedcfd2-6145-479e-b6a7-5606ef2ec458" alt="Resim" width="500" />
</div>


As seen in the graph, NLP is widely employed in the BFSI sector, particularly for automating customer interactions, detecting fraud, and analyzing sentiment. Its extensive use highlights its pivotal role in improving operational efficiency, customer satisfaction, and data-driven decision-making in the finance industry.
[Picture: johnsnowlabs][3]

[3]: https://www.johnsnowlabs.com/examining-the-impact-of-nlp-in-financial-services/
> [!TIP]
> For financial institutions, which can be reluctant to deploy cutting-edge techniques like machine learning, this socialization process is an important step. “As more and more people see it work and understand the lingo, they see that it’s not a dark art — it’s math."
# Where is NLP Going?
## Investments in NLP 📈
Investments in Natural Language Processing (NLP) are on the rise as businesses recognize its potential benefits and integrate it across various industries. Major players like Microsoft have made significant investments in advanced NLP, indicating its importance in healthcare, finance, advertising, and customer service. As NLP evolves, transitioning from human-computer interaction to conversation, service desks are now providing more personalized responses using conversational AI.
## Companies will use NLG 🏢
Additionally, Natural Language Generation (NLG) is gaining popularity as companies experiment with automating text generation for reports and descriptions, saving time and improving data analysis. Sentiment analysis, another aspect of NLP, is being widely implemented across sectors to assess customer sentiments and feedback. Voice biometrics is becoming more common for identity verification, especially in call centers and healthcare.
## Humanoid Robotics 🤖
Humanoid Robotics, combining robotics and NLP, is an exciting field with robots programmed to interact using natural language. This technology is particularly valuable in healthcare, where robots can communicate, answer questions, and provide emotional support. As NLP and machine learning advance, humanoid robots are becoming more intelligent, changing the way we work, learn, and interact with technology across various industries.

# Ethical approach and challenges
## Fairness
Bias in AI, especially in Natural Language Processing (NLP), is a significant challenge. To address this, there are ongoing efforts to teach AI models using diverse datasets, aiming to make language tools fair and inclusive. This is especially important in critical areas like law enforcement and human resources.

For instance, in hiring processes, AI tools are being developed to evaluate candidates' qualifications without any bias related to gender, ethnicity, or background. Likewise, law enforcement is adopting AI systems trained on diverse datasets to ensure unbiased communication with the public. These steps toward ethical AI show a commitment to creating a fairer technology environment. This focus on ethical AI is crucial in social media too, where algorithms now better identify and reduce biased content, promoting a healthier and more respectful online space for diverse global communities.
## Privacy
NLP systems require vast amounts of data to function, but collecting and using this data can raise serious concerns about privacy. In 2023, we can expect to see an increased focus on data privacy in NLP, with new regulations and best practices being developed to protect user data.
## Explainability
As NLP models become more complex, it can be difficult to understand how they arrive at certain decisions. In 2023, it will be important for NLP developers to focus on creating models that are explainable, making it easier to understand how the model arrived at a particular decision.

