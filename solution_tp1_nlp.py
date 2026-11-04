"""Solution complète du TP1 de prétraitement NLP (Reuters/NLTK).

Exécuter dans un environnement ayant les dépendances et ressources listées dans le PDF.
Le script affiche les résultats des exercices et enregistre les graphiques dans ./figures_tp1.
"""
import html, re, string, unicodedata, random
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import reuters, words as nltk_words, stopwords, wordnet
from nltk.tokenize import (word_tokenize, TreebankWordTokenizer,
                           WordPunctTokenizer, RegexpTokenizer)
from nltk.probability import FreqDist
from nltk.stem import (WordNetLemmatizer, PorterStemmer, LancasterStemmer,
                       SnowballStemmer)
from nltk import pos_tag, RegexpParser, ne_chunk
from wordcloud import WordCloud

random.seed(42); np.random.seed(42)
for resource in ['reuters','punkt','punkt_tab','stopwords','wordnet','words',
                 'averaged_perceptron_tagger','averaged_perceptron_tagger_eng',
                 'maxent_ne_chunker','maxent_ne_chunker_tab']:
    nltk.download(resource, quiet=True)

FILEID = 'test/14826'
text = reuters.raw(FILEID)
english_vocab = {w.lower() for w in nltk_words.words()}
financial_terms = {'fed','rate','percent','bank','stock','market','price','trade',
                   'export','import','deficit','billion','million','share','bond'}
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return re.sub(r' {2,}', ' ', re.sub(r'[\n\t\r]+', ' ', text)).strip()

SUFFIXES_CONTRACTION = {"'s", "'re", "'ve", "'ll", "'d", "'m"}
RACINES_IRREGULIERES = {'ca':'can', 'wo':'will', 'sha':'shall'}
def fix_contractions(tokens):
    out=[]; i=0
    while i < len(tokens):
        nxt=tokens[i+1].lower() if i+1<len(tokens) else ''
        if nxt == "n't":
            out.extend([RACINES_IRREGULIERES.get(tokens[i].lower(), tokens[i]), 'not']); i+=2
        elif tokens[i].lower() in SUFFIXES_CONTRACTION: i+=1
        else: out.append(tokens[i]); i+=1
    return out

def normalize(tokens):
    tokens=fix_contractions(tokens)
    tokens=[unicodedata.normalize('NFKD',t.lower()).encode('ascii','ignore').decode() for t in tokens]
    punct=set(string.punctuation)|{'$','%','&'}
    return [t for t in tokens if t not in punct and t not in stop_words and t.isalpha() and len(t)>=2]

def normalize_financier(tokens):
    """Conserve les valeurs sous forme de marqueurs; traite les séquences de tokens."""
    out=[]; i=0
    while i < len(tokens):
        t=tokens[i].lower()
        # Pourcentages word_tokenize: nombre suivi du signe %, ou suffixe pct.
        num=re.fullmatch(r'\d+(?:\.\d+)?', t)
        nxt=tokens[i+1].lower() if i+1<len(tokens) else ''
        if num and (nxt=='%' or nxt=='pct'):
            out.append('TAUX'); i+=2; continue
        # Montants avec symbole dollar séparé ou montant suivi d'un suffixe/indicateur.
        amount=t.lstrip('$')
        m=re.fullmatch(r'(\d+(?:\.\d+)?)(m|mln|million|bn|bln|billion)?', amount)
        if t=='$' and i+1<len(tokens):
            m2=re.fullmatch(r'(\d+(?:\.\d+)?)(m|mln|million|bn|bln|billion)?',tokens[i+1].lower())
            if m2:
                suf=m2.group(2); out.append('MONTANT_MILLIARD' if suf in {'bn','bln','billion'} else 'MONTANT_MILLION' if suf in {'m','mln','million'} else 'MONTANT')
                i+=2
                if i<len(tokens) and tokens[i].lower() in {'dlr','dlrs'}: i+=1
                continue
        if m and (m.group(2) or nxt in {'dlr','dlrs','m','mln','million','bn','bln','billion'}):
            suf=m.group(2) or nxt
            out.append('MONTANT_MILLIARD' if suf in {'bn','bln','billion'} else 'MONTANT_MILLION' if suf in {'m','mln','million'} else 'MONTANT')
            i+=1
            if not m.group(2) and i<len(tokens) and tokens[i].lower() in {'dlr','dlrs','m','mln','million','bn','bln','billion'}: i+=1
            if i<len(tokens) and tokens[i].lower() in {'dlr','dlrs'}: i+=1
            continue
        if re.fullmatch(r'(19|20)\d{2}',t): out.append('ANNEE'); i+=1; continue
        if t in {'dlr','dlrs','pct','m','mln','million','bn','bln','billion'}: i+=1; continue
        out.append(tokens[i]); i+=1
    # Les marqueurs sont protégés de normalize(); normaliser les autres séparément.
    result=[]
    for x in out:
        if x in {'TAUX','MONTANT','MONTANT_MILLION','MONTANT_MILLIARD','ANNEE'}: result.append(x)
        else: result.extend(normalize([x]))
    return result

text_clean=clean_text(text)
tokens_0=word_tokenize(text_clean)
tokens_4=normalize(tokens_0)

print('Article',FILEID,'| catégories:',reuters.categories(FILEID),'| caractères:',len(text))
print('\nSECTION 3 - Comparaison tokeniseurs')
custom_tokenizer=RegexpTokenizer(r"\$?\d+(?:\.\d+)?(?:bn|bln|billion|mln|million|m)?|(?:[A-Za-z]\.){2,}|\d+(?:\.\d+)?%|\w+|[^\w\s]+")
tokenizers={'word_tokenize':word_tokenize(text),'TreebankWordTokenizer':TreebankWordTokenizer().tokenize(text),
            'WordPunctTokenizer':WordPunctTokenizer().tokenize(text),'RegexpTokenizer':RegexpTokenizer(r'\w+').tokenize(text),
            'custom_tokenizer':custom_tokenizer.tokenize(text)}
rows=[]
for name,toks in tokenizers.items():
    vocab={t.lower() for t in toks}
    rows.append({'Tokeniseur':name,'Nb tokens':len(toks),'Taille vocabulaire':len(vocab),
                 'Termes financiers /15':len(financial_terms&vocab),
                 'Couverture lexicale approx. (%)':round(len(vocab&english_vocab)/len(vocab)*100,1)})
print(pd.DataFrame(rows).to_string(index=False))
examples=['U.S.','$1.2bn','2.5%','31/12/2024','New York-based']
qual={}
for e in examples:
    qual[e]={}
    for n in tokenizers:
        if n=='custom_tokenizer': qual[e][n]=str(custom_tokenizer.tokenize(e))
        else:
            fn={'word_tokenize':word_tokenize,'TreebankWordTokenizer':TreebankWordTokenizer().tokenize,
                'WordPunctTokenizer':WordPunctTokenizer().tokenize,'RegexpTokenizer':RegexpTokenizer(r'\w+').tokenize}[n]
            qual[e][n]=str(fn(e))
print(pd.DataFrame([{'Expression':e,**v} for e,v in qual.items()]).to_string(index=False))
oov=[t for t in word_tokenize(text) if t.lower() not in english_vocab]
print('20 tokens hors lexique:',oov[:20])

print('\nSECTION 4 - Normalisation')
tokens_brut=word_tokenize(text); tokens_nettoyes=word_tokenize(text_clean)
stages={'Brut':tokens_brut,'Nettoyé':tokens_nettoyes,'Normalisé':tokens_4}
v0=len(set(tokens_brut))
print(pd.DataFrame([{'Étape':k,'Nb tokens':len(v),'Vocab unique':len(set(v)),
                     'Réduction vocab (%)':round(100*(1-len(set(v))/v0),1),
                     'Réduction tokens (%)':round(100*(1-len(v)/len(tokens_brut)),1)} for k,v in stages.items()]).to_string(index=False))
steps=[]; x=tokens_0
def record(label, seq): steps.append({'Étape':label,'Tokens':len(seq),'Vocabulaire':len(set(seq))})
record('Départ',x); x=fix_contractions(x); record('Contractions',x)
x=[t.lower() for t in x]; record('Minuscules',x)
x=[unicodedata.normalize('NFKD',t).encode('ascii','ignore').decode() for t in x]; record('Unicode ASCII',x)
x=[t for t in x if t not in (set(string.punctuation)|{'$','%','&'})]; record('Ponctuation',x)
x=[t for t in x if t not in stop_words]; record('Mots vides',x)
x=[t for t in x if t.isalpha()]; record('Alphabétique',x)
x=[t for t in x if len(t)>=2]; record('Longueur >= 2',x)
print(pd.DataFrame(steps).to_string(index=False))
fin_tokens=normalize_financier(tokens_0)
print('normalisé général:',len(tokens_4),'tokens /',len(set(tokens_4)),'types; financier:',len(fin_tokens),'/',len(set(fin_tokens)))
print('Phrase financière:',normalize_financier(word_tokenize('Profit rose 2.5% to $1.2bn, or 3.1 pct, up from $500m in 1986.')))
phrase="The U.S. can't raise rates; Japan's banks didn't sell."
naive=re.sub(r"'(s|t|re|ve|ll|d|m)\b",'',phrase)
print('Contractions naïves:',word_tokenize(naive)); print('Contractions après tokenisation:',fix_contractions(word_tokenize(phrase)))

print('\nSECTION 5 - Stemming / lemmatisation')
lemmatizer=WordNetLemmatizer(); porter=PorterStemmer(); lancaster=LancasterStemmer(); snowball=SnowballStemmer('english')
test_words=['acquiring','acquired','banks','falling','earnings','better','traded','prices','companies','rising']
print(pd.DataFrame([{'Mot':w,'Lemme sans POS':lemmatizer.lemmatize(w),'Porter':porter.stem(w),'Lancaster':lancaster.stem(w),'Snowball':snowball.stem(w)} for w in test_words]).to_string(index=False))
techniques={'Normalisé':tokens_4,'Lemme sans POS':[lemmatizer.lemmatize(t) for t in tokens_4],
            'Porter':[porter.stem(t) for t in tokens_4],'Lancaster':[lancaster.stem(t) for t in tokens_4],
            'Snowball':[snowball.stem(t) for t in tokens_4]}
print(pd.DataFrame([{'Technique':k,'Vocabulaire':len(set(v)),'Réduction (%)':round(100*(1-len(set(v))/len(set(tokens_4))),1),'Couverture lexicale (%)':round(100*len(set(v)&english_vocab)/len(set(v)),1)} for k,v in techniques.items()]).to_string(index=False))

print('\nSECTION 6 - POS et lemmatisation')
def get_wordnet_pos(tag): return {'J':wordnet.ADJ,'V':wordnet.VERB,'N':wordnet.NOUN,'R':wordnet.ADV}.get(tag[0],wordnet.NOUN)
def pos_tag_en_contexte(tokens):
    tagged=pos_tag(fix_contractions(tokens))
    return [(n[0],tag) for tok,tag in tagged for n in [normalize([tok])] if n]
def pos_features(tags):
    c=Counter(tag for _,tag in tags); total=max(sum(1 for _,tag in tags if tag[0].isalpha()),1)
    return {'ratio_noms':sum(c[t] for t in {'NN','NNS','NNP','NNPS'})/total,
            'ratio_verbes':sum(c[t] for t in {'VB','VBD','VBG','VBN','VBP','VBZ'})/total,
            'ratio_adjectifs':sum(c[t] for t in {'JJ','JJR','JJS'})/total,
            'ratio_noms_propres':(c['NNP']+c['NNPS'])/total}
pos_tags=pos_tag_en_contexte(tokens_0); pos_tags_ctx=pos_tag(tokens_0)
assert [t for t,_ in pos_tags]==tokens_4
lemmas_no=[lemmatizer.lemmatize(t) for t in tokens_4]
tokens_lemma=[lemmatizer.lemmatize(t,get_wordnet_pos(p)) for t,p in pos_tags]
modified=[(t,p,a,b) for (t,p),a,b in zip(pos_tags,lemmas_no,tokens_lemma) if a!=b]
print('Changements POS:',len(modified),'/',len(tokens_4),'| par tag:',Counter(p for _,p,_,_ in modified))
print(pd.DataFrame([{'Token':t,'POS':p,'Sans POS':a,'Avec POS':b} for t,p,a,b in modified]).to_string(index=False))
untagged=pos_tag(tokens_4)
context_by_token={t:p for t,p in pos_tags}
diff=[]
for t,p in untagged:
    if t in context_by_token and p!=context_by_token[t] and len(diff)<20:
        diff.append((t,p,context_by_token[t],lemmatizer.lemmatize(t,get_wordnet_pos(p)),lemmatizer.lemmatize(t,get_wordnet_pos(context_by_token[t]))))
print('POS hors/en contexte (premières différences):',diff[:10])
print('Features article principal:',pos_features(pos_tags_ctx))
article_features=[]
for fid in ['test/14826','test/14828']:
    tx=clean_text(reuters.raw(fid)); tags=pos_tag(word_tokenize(tx)); article_features.append({'id':fid,'categories':', '.join(reuters.categories(fid)),**pos_features(tags)})
print(pd.DataFrame(article_features).to_string(index=False))

print('\nSECTION 7 - Chunking et NER')
tree=RegexpParser(r'NP: {<DT>?<JJ>*<NN.*>+}').parse(pos_tags_ctx)
chunks_np=[' '.join(w for w,_ in st.leaves()) for st in tree.subtrees() if st.label()=='NP']
ner_tree=ne_chunk(pos_tags_ctx,binary=False)
def nltk_entities(tree):
    d=defaultdict(list)
    for st in tree:
        if hasattr(st,'label'): d[st.label()].append(' '.join(w for w,_ in st.leaves()))
    return d
entities=nltk_entities(ner_tree)
print('Chunks NP:',len(chunks_np),'Top:',Counter(chunks_np).most_common(15))
print('Entités NLTK:',{k:Counter(v).most_common(8) for k,v in entities.items()})
print('NER normalisée:',sum(hasattr(st,'label') for st in ne_chunk(pos_tags,binary=False)),'vs contexte:',sum(map(len,entities.values())))

# Vérité terrain explicite, mention par mention, sur les trois premières phrases/paragraphe du corps.
# L'extrait d'évaluation est défini sans le titre; les mentions sont alignées sur les mots présents.
gold=[('U.S.','lieu'),('Japan','lieu'),('Asia','lieu'),
      ('U.S.','lieu'),('Japan','lieu'),('U.S.','lieu'),('Tokyo','lieu')]
def canonical(label, system):
    if label=='PERSON': return 'personne'
    if label in ({'ORGANIZATION'} if system=='nltk' else {'ORG'}): return 'organisation'
    if label in ({'GPE','LOCATION','GSP'} if system=='nltk' else {'GPE','LOC'}): return 'lieu'
    return None
def eval_mentions(preds,gold):
    remaining=Counter(gold); correct=0; kept=0
    for ent,label,system in preds:
        cat=canonical(label,system)
        if cat is None: continue
        kept+=1
        key=(ent,cat)
        if remaining[key]: remaining[key]-=1; correct+=1
    total=sum(Counter(gold).values())
    return {'correctes':correct,'détections retenues':kept,'mentions annotées':total,
            'précision':correct/kept if kept else 0,'rappel':correct/total if total else 0}

try:
    import spacy
    nlp_spacy=spacy.load('en_core_web_sm')
    doc=nlp_spacy(text_clean)
    print('spaCy entités:',Counter(e.label_ for e in doc.ents))
    # Pour l'évaluation courte, le titre est exclu et les 3 premiers paragraphes textuels retenus.
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    excerpt=' '.join(lines[1:8])
    doc_short=nlp_spacy(excerpt)
    sp_preds=[(e.text,e.label_,'spacy') for e in doc_short.ents]
    # NLTK sur le même extrait.
    nltk_short=nltk_entities(ne_chunk(pos_tag(word_tokenize(excerpt)),binary=False))
    nl_preds=[(e,l,'nltk') for l,es in nltk_short.items() for e in es]
    print('Évaluation NER selon gold explicite:',pd.DataFrame([{'Système':'NLTK',**eval_mentions(nl_preds,gold)},{'Système':'spaCy',**eval_mentions(sp_preds,gold)}]).to_string(index=False))
except (ImportError,OSError) as err:
    print('spaCy non disponible:',err,'; mesures NLTK restent calculées.')

print('Second article entities:',{k:len(v) for k,v in nltk_entities(ne_chunk(pos_tag(word_tokenize(clean_text(reuters.raw('test/14828')))),binary=False)).items()})

print('\nSECTION 8 - Évolution du vocabulaire')
states={'Brut':tokens_brut,'Normalisé':tokens_4,'Lemmatisé':tokens_lemma}
print(pd.DataFrame([{'Étape':k,'Vocabulaire':len(set(v)),'Top 5':Counter(v).most_common(5)} for k,v in states.items()]).to_string(index=False))
outdir=Path('figures_tp1'); outdir.mkdir(exist_ok=True)
plt.figure(figsize=(8,4)); plt.bar(states.keys(),[len(set(v)) for v in states.values()],color='steelblue'); plt.ylabel('Tokens uniques'); plt.title('Taille du vocabulaire'); plt.tight_layout(); plt.savefig(outdir/'vocabulaire.png',dpi=160); plt.close()
fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,(label,toks) in zip(axes,states.items()):
    ax.imshow(WordCloud(width=400,height=300,background_color='white').generate_from_frequencies(Counter(toks))); ax.axis('off'); ax.set_title(label)
fig.tight_layout(); fig.savefig(outdir/'nuages.png',dpi=160); plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,(label,toks) in zip(axes,states.items()):
    pairs=Counter(toks).most_common(15); ax.barh([x[0] for x in pairs[::-1]],[x[1] for x in pairs[::-1]],color='steelblue'); ax.set_title(label)
fig.tight_layout(); fig.savefig(outdir/'frequences.png',dpi=160); plt.close(fig)
print('Graphiques écrits dans',outdir.resolve())

print('\nSECTION 9 - Pipeline intégré')
def nlp_pipeline(text,with_chunks=True,with_ner=True,verbose=True):
    tc=clean_text(text); t0=word_tokenize(tc); ctx=pos_tag(t0); t4=normalize(t0); pt=pos_tag_en_contexte(t0)
    lemma=[lemmatizer.lemmatize(t,get_wordnet_pos(p)) for t,p in pt]
    chunks=None; ents=None
    if with_chunks:
        tree=RegexpParser(r'NP: {<DT>?<JJ>*<NN.*>+}').parse(ctx)
        chunks=[' '.join(w for w,_ in st.leaves()) for st in tree.subtrees() if st.label()=='NP']
    if with_ner: ents=dict(nltk_entities(ne_chunk(ctx,binary=False)))
    if verbose:
        for label,seq in [('tokens_0',t0),('tokens_4',t4),('pos_tags',pt),('tokens_lemma',lemma)]:
            print(f'{label:14} tokens={len(seq):4} vocabulaire={len(set(seq)):4}')
    return {'text_clean':tc,'tokens_0':t0,'tokens_4':t4,'pos_tags':pt,'pos_tags_ctx':ctx,
            'tokens_lemma':lemma,'chunks_np':chunks,'entities':ents}

for fid in ['test/14826','test/14828','test/14829']:
    print('\n',fid,reuters.categories(fid)); result=nlp_pipeline(reuters.raw(fid),verbose=True)
