"""TP2 - Vectorisation et word embeddings pré-entraînés."""
import html, re, string, unicodedata, time
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import reuters, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

for resource in ['reuters', 'punkt', 'punkt_tab', 'stopwords']:
    nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return re.sub(r' {2,}', ' ', re.sub(r'[\n\t\r]+', ' ', text)).strip()

SUFFIXES_CONTRACTION = {"'s", "'re", "'ve", "'ll", "'d", "'m"}
IRREGULAR = {'ca': 'can', 'wo': 'will', 'sha': 'shall'}
def fix_contractions(tokens):
    result=[]; i=0
    while i < len(tokens):
        following=tokens[i+1].lower() if i+1<len(tokens) else ''
        if following == "n't":
            result.extend([IRREGULAR.get(tokens[i].lower(),tokens[i]),'not']); i+=2
        elif tokens[i].lower() in SUFFIXES_CONTRACTION: i+=1
        else: result.append(tokens[i]); i+=1
    return result

def normalize(tokens):
    tokens=fix_contractions(tokens)
    tokens=[unicodedata.normalize('NFKD',t.lower()).encode('ascii','ignore').decode() for t in tokens]
    punctuation=set(string.punctuation)|{'$','%','&'}
    return [t for t in tokens if t not in punctuation and t not in stop_words and t.isalpha() and len(t)>=2]

texte_art=reuters.raw(fileids=['test/14826'])
def phrases_normalisees(texte):
    phrases=[normalize(word_tokenize(s)) for s in sent_tokenize(clean_text(texte))]
    return [' '.join(p) for p in phrases if p]
def phrases_tokenisees(texte):
    phrases=[]
    for s in sent_tokenize(clean_text(texte)):
        toks=[t for t in word_tokenize(s) if t.isalpha()]
        if toks: phrases.append(toks)
    return phrases

textes_art=phrases_normalisees(texte_art)
phrases_art=phrases_tokenisees(texte_art)
print('Phrases:',len(textes_art),'| premier document normalisé:',textes_art[0])
print('\nSECTION 1 - Préparer l’article')
print('Normalisé [1]:',textes_art[1])
print('Tokenisé [1]:',' '.join(phrases_art[1]))

print('\nSECTION 2 - One-hot')
from sklearn.metrics.pairwise import cosine_similarity
tokens_art=[t for doc in textes_art for t in doc.split()]
vocab_art=sorted(set(tokens_art)); index={mot:i for i,mot in enumerate(vocab_art)}
def one_hot(mot,index):
    v=np.zeros(len(index),dtype=np.int8); v[index[mot]]=1; return v
M=np.stack([one_hot(t,index) for t in tokens_art])
print('Matrice one-hot:',M.shape,'| taux de remplissage:',round(float(M.mean()),6))
print('cos(trade, tariffs)=',cosine_similarity([one_hot('trade',index)],[one_hot('tariffs',index)])[0,0])
print('cos(trade, japan)=',cosine_similarity([one_hot('trade',index)],[one_hot('japan',index)])[0,0])
try: one_hot('wheat',index)
except KeyError as exc: print('wheat absent: KeyError',exc)

print('\nSECTION 3 - Bag of Words')
from sklearn.feature_extraction.text import CountVectorizer
param=dict(tokenizer=str.split,token_pattern=None,lowercase=False)
cv=CountVectorizer(**param); X_bow=cv.fit_transform(textes_art); noms=cv.get_feature_names_out()
print('Dimension:',X_bow.shape,'| densité:',round(X_bow.nnz/np.prod(X_bow.shape),4))
termes=['trade','japan','tariffs','exports','said']
print(pd.DataFrame(X_bow[:6].toarray(),columns=noms)[termes].to_string(index=False))
totaux=np.asarray(X_bow.sum(axis=0)).ravel()
print('Top 8 termes:',list(zip(noms[totaux.argsort()[-8:][::-1]],totaux[totaux.argsort()[-8:][::-1]])))
queries=cv.transform(['japan exports tariffs','tariffs exports japan'])
print('Vecteurs des permutations identiques:',(queries[0]!=queries[1]).nnz==0)
texte_14828=reuters.raw(fileids=['test/14828']); textes_14828=phrases_normalisees(texte_14828)
tok_14828=[t for d in textes_14828 for t in d.split()]
oov_bow=[t for t in tok_14828 if t not in cv.vocabulary_]
print('OOV article 14828:',len(oov_bow),'/',len(tok_14828),'=',round(len(oov_bow)/max(len(tok_14828),1),3),'| occurrences non nulles après transform:',cv.transform(textes_14828).nnz)

print('\nSECTION 4 - TF-IDF')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(**param); X_tfidf=tfidf.fit_transform(textes_art); noms_tfidf=tfidf.get_feature_names_out()
print(pd.DataFrame(X_tfidf[:6].toarray(),columns=noms_tfidf)[termes].round(2).to_string(index=False))
idf=tfidf.idf_; order=idf.argsort()
print('IDF faible:',[(noms_tfidf[i],round(idf[i],2)) for i in order[:4]])
print('IDF élevé:',[(noms_tfidf[i],round(idf[i],2)) for i in order[-3:]])
said_i=tfidf.vocabulary_.get('said'); print('IDF said:',round(idf[said_i],3) if said_i is not None else 'absent')
print('Termes avec IDF maximal:',int(np.isclose(idf,idf.max()).sum()))
sims=cosine_similarity(X_tfidf); np.fill_diagonal(sims,0)
i,j=np.unravel_index(sims.argmax(),sims.shape)
print(f'Phrases les plus proches (cos={sims[i,j]:.3f}):\n- {textes_art[i]}\n- {textes_art[j]}')
paires=sims[np.triu_indices_from(sims,1)]
zero=np.argwhere(np.triu(sims==0,1))
print('Part de paires à cosinus nul:',round(float((paires==0).mean()),3))
print('Exemple de paire à cosinus nul:',(textes_art[zero[0,0]],textes_art[zero[0,1]]) if len(zero) else 'aucune')

print('\nSECTION 5 - N-grammes')
rows=[]
for n in (1,2,3):
    X=TfidfVectorizer(ngram_range=(1,n),**param).fit_transform(textes_art)
    rows.append({'ngram_range':f'(1, {n})','Dimension':X.shape[1],'Valeurs non nulles':X.nnz,'Zéros (%)':round(100*(1-X.nnz/np.prod(X.shape)),2)})
print(pd.DataFrame(rows).to_string(index=False))
tfidf_ng=TfidfVectorizer(ngram_range=(1,2),**param); X_ng=tfidf_ng.fit_transform(textes_art); noms_ng=tfidf_ng.get_feature_names_out()
ligne=X_ng[1].toarray().ravel(); print('Top termes pour phrase [1]:',[(noms_ng[k],round(ligne[k],3)) for k in ligne.argsort()[-6:][::-1]])
bigram_docfreq=Counter(bg for doc in textes_art for bg in set(zip(doc.split(),doc.split()[1:])))
print('Bigrammes présents dans une seule phrase:',sum(freq==1 for freq in bigram_docfreq.values()))
bank_japan=' '.join(normalize(word_tokenize('bank of Japan')))
print('Normalisation de "bank of Japan":',bank_japan,'| bigramme présent dans l’article ?',bank_japan in noms_ng)

print('\nSECTION 6 - Word2Vec (téléchargement ~1,7 Go au premier lancement)')
print('Cette cellule est volontairement séparée : exécutez-la seulement si le téléchargement et ~0,5 Go de RAM sont acceptables.')
# import gensim.downloader as api
# from gensim.models import KeyedVectors
# chemin = api.load('word2vec-google-news-300', return_path=True)
# debut=time.time(); w2v=KeyedVectors.load_word2vec_format(chemin,binary=True,limit=300000)
# print('Vocabulaire chargé:',len(w2v),'| dimension:',w2v.vector_size,'| secondes:',round(time.time()-debut))
# def top(wv,mot,k=6): return [w for w,_ in wv.most_similar(mot,topn=k)]
# for mot in ['Japan','japan','crude_oil','the','of','and','dlrs']:
#     print(mot,'présent' if mot in w2v else 'absent')
# print(pd.DataFrame([{'Mot':m,'Voisins Word2Vec':', '.join(top(w2v,m,5))} for m in ['oil','Oil','profit','dollar','wheat','tariffs','Japan','japan']]).to_string(index=False))

print('\nSECTION 7 - GloVe (~130 Mo)')
print('À lancer après l’installation de gensim et le chargement Word2Vec.')
# glove=api.load('glove-wiki-gigaword-100')
# vocab_art_token=sorted({t for s in phrases_art for t in s})
# print('Vocabulaire GloVe:',len(glove),'| dimension:',glove.vector_size)
# print('Absents de Word2Vec:',sum(t not in w2v for t in vocab_art_token),'| absents de GloVe:',sum(t.lower() not in glove for t in vocab_art_token))
# print(pd.DataFrame([{'Mot':m,'Word2Vec':', '.join(top(w2v,m,4)),'GloVe':', '.join(top(glove,m.lower(),4))} for m in ['oil','crude','shares','tender','bank']]).to_string(index=False))

print('\nSECTION 8 - FastText compressé (~240 Mo)')
print('À lancer seulement si les dépendances et le téléchargement externe sont disponibles.')
# import os, urllib.request, compress_fasttext
# FICHIER='ft_cc.en.300_freqprune_400K_100K_pq_300.bin'
# URL='https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/'+FICHIER
# if not os.path.exists(FICHIER): urllib.request.urlretrieve(URL,FICHIER)
# debut=time.time(); fasttext=compress_fasttext.models.CompressedFastTextKeyedVectors.load(FICHIER)
# print('Vocabulaire:',len(fasttext),'| dimension:',fasttext.vector_size,'| secondes:',round(time.time()-debut))
# def voisins_ft(mot,k=3): return [w for w in top(fasttext,mot,30) if len(w)<=20][:k]
# for mot in ['tarifs','dolars','oilprices']:
#     print(mot,'FastText:',mot in fasttext.key_to_index,'Word2Vec:',mot in w2v,'GloVe:',mot in glove,voisins_ft(mot))
# print('cos(tarifs, tariffs)=',round(float(fasttext.similarity('tarifs','tariffs')),2))
# print('cos(tarifs, wheat)=',round(float(fasttext.similarity('tarifs','wheat')),2))
# print("zzqxqz in fasttext:",'zzqxqz' in fasttext,'| dans le vocabulaire:', 'zzqxqz' in fasttext.key_to_index)
# for mot in ['heavily','exports']:
#     print(mot,'Word2Vec:',top(w2v,mot,5),'| FastText:',voisins_ft(mot,5))

print('\nSECTION 9 - Comparaison des modèles')
print('Les valeurs numériques dépendent des modèles chargés; la cellule suivante est décommentable après les sections 6 à 8.')
# print(pd.DataFrame([{'Mot':m,'Word2Vec':', '.join(top(w2v,m,3)),'GloVe':', '.join(top(glove,m.lower(),3)),'FastText':', '.join(voisins_ft(m))} for m in ['oil','bank','tariffs','profit']]).to_string(index=False))
# for a,b in [('oil','crude'),('oil','wheat')]:
#     print(f'{a}-{b}: Word2Vec {w2v.similarity(a,b):.2f} | GloVe {glove.similarity(a,b):.2f} | FastText {fasttext.similarity(a,b):.2f}')
