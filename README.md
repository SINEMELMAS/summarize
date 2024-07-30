# summarize
import re
import spacy
from spacy import displacy
from collections import Counter
from transformers import pipeline #to summarize
from transformers import T5Tokenizer, T5ForConditionalGeneration

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

nlp = spacy.load('en_core_web_sm')

text=input("Enter a text: ")
doc = nlp(text)

entities = list(set([(ent.text, ent.label_) for ent in doc.ents]))

print("Named Entities:")
for entity in entities:
    print(f"{entity[0]} ({entity[1]})")

def sentences(text):
    sent_token = list(doc.sents)
    print(sent_token)

sentences(text)

words = [token.text for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
print(words)


summary = summarizer(text, max_length=100, min_length=30, do_sample=False)

print(summary[0]['summary_text'])

freq_word= Counter(words)
keywords = freq_word.most_common(5)
print(f"Keywords: {keywords}")
