import spacy

# must run `python3 -m spacy download en_core_web_sm` to install models
nlp = spacy.load("en_core_web_sm")
# large English: nlp = spacy.load("en_core_web_lg")
# Spanish: nlp = spacy.load("es_core_web_sm") 
# Chinese: nlp = spacy.load("zh_core_web_sm")
# no vietnamese support for advanced features

print('========================================')
print('Sentence segmentation:')
doc = nlp("This is a sentence... This is another sentence. This sentence mentions the U.K. and the number 3.14159")
for sent in doc.sents:
    print('  ',sent.text)

print('========================================')
doc = nlp('Autonomous cars shift insurance liability toward manufacturers, and new york city doesn\'t like that')
print('Verbs:')
for chunk in doc:
    if chunk.pos == spacy.symbols.VERB:
        print('  ', chunk.text)

print('Nouns:')
for chunk in doc:
    if chunk.pos == spacy.symbols.NOUN:
        print('  ', chunk.text)

print('Noun chunks:')
for chunk in doc.noun_chunks:
    print('  ', chunk.text)

print('========================================')
print('Named entity recognition (NER):')
doc = nlp('Apple is looking at buying U.K. startup for $1 billion, and San Francisco is considering banning sidewalk delivery robots, and The Killers is an awesome band.')
for ent in doc.ents:
    print('  ', ent.text, ent.start_char, ent.end_char, ent.label_)

spacy.displacy.serve(doc, style="ent")
