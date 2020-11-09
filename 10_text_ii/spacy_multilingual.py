print('========================================')
print('English')
import spacy.lang.en
en = spacy.lang.en.English()
doc = en('I love data mining')
for token in doc:
    print('  ',token.lemma_,token.pos)

print('========================================')
print('Spanish')
import spacy.lang.es
es = spacy.lang.es.Spanish()
doc = es('Me encanta la minería de datos')
for token in doc:
    print(token.lemma_)

print('========================================')
print('Chinese')
import spacy.lang.zh
zh = spacy.lang.zh.Chinese()
doc = zh('我喜歡數據挖掘')
for token in doc:
    print(token.lemma_)

print('========================================')
print('Vietnamese')
import spacy.lang.vi
vi = spacy.lang.vi.Vietnamese()
doc = vi('Tôi thích khai thác dữ liệu')
for token in doc:
    print(token.lemma_)

