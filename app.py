import streamlit as st
import stanza
import pandas as pd
import requests
import urllib.parse
import unicodedata
from transformers import pipeline

st.set_page_config(
    page_title="Ancient Greek Engine: Wordnet & Sentimento",
    page_icon="🏛️",
    layout="wide"
)

# Mapeamento explícito de rótulos de sentimento
LABEL_MAP = {
    "LABEL_0": "LABEL_0 (Negativo)",
    "LABEL_1": "LABEL_1 (Positivo)",
    "LABEL_2": "LABEL_2 (Neutro)",
    "NEGATIVE": "NEGATIVE (Negativo)",
    "POSITIVE": "POSITIVE (Positivo)"
}

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO E CACHE DOS MODELOS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_stanza():
    stanza.download('grc', verbose=False)
    return stanza.Pipeline('grc', processors='tokenize,mwt,pos,lemma', verbose=False)

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="nlpaueb/bert-base-greek-uncased-v1",
        top_k=None
    )

with st.spinner("Carregando modelos de NLP em Grego Antigo..."):
    stanza_nlp = load_stanza()
    sentiment_nlp = load_sentiment_model()

# -----------------------------------------------------------------------------
# 2. CONSULTA ROBUSTA À API DO ANCIENT GREEK WORDNET
# -----------------------------------------------------------------------------
def remove_diacritics(text):
    """Remove acentos e espíritos do grego para busca de fallback."""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def query_agwn_endpoint(lemma_str):
    """Realiza a requisição HTTP pura ao servidor do AGWN."""
    encoded = urllib.parse.quote(lemma_str)
    url = f"https://greekwordnet.chs.harvard.edu/api/lemma/{encoded}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            items = data if isinstance(data, list) else data.get("synsets", [])
            synsets_found = []
            for item in items:
                s_id = item.get("synset_id", item.get("id", ""))
                gloss = item.get("gloss", item.get("definition", ""))
                pos = item.get("pos", "")
                synsets_found.append(f"[{s_id}] ({pos}) {gloss}")
            return " | ".join(synsets_found) if synsets_found else None
    except Exception:
        pass
    return None

def get_agwn_synsets(lemma):
    """Tenta buscar com acentos; se falhar, tenta sem acentos/diacríticos."""
    lemma_clean = lemma.strip().lower()
    
    # 1ª Tentativa: Lema original do Stanza
    res = query_agwn_endpoint(lemma_clean)
    if res:
        return res
    
    # 2ª Tentativa: Lema sem diacríticos (acentos/espíritos)
    lemma_no_accents = remove_diacritics(lemma_clean)
    if lemma_no_accents != lemma_clean:
        res = query_agwn_endpoint(lemma_no_accents)
        if res:
            return res
            
    return "Nenhum synset retornado"

# -----------------------------------------------------------------------------
# 3. INTERFACE PRINCIPAL
# -----------------------------------------------------------------------------
st.title("🏛️ Ancient Greek Engine: AGWN API & Sentiment")

default_text = "οὐ γὰρ ἦν ἄλλως χρήσασθαι τῇ κατὰ ταῦτα γυμνασίᾳ ."
text_input = st.text_area("Insira o texto em Grego Antigo:", value=default_text, height=120)

if st.button("🚀 Processar Texto", type="primary"):
    if not text_input.strip():
        st.warning("Por favor, insira um texto em Grego Antigo.")
    else:
        # A) PROCESSAMENTO MORFOLÓGICO & CONSULTA AGWN
        doc = stanza_nlp(text_input)
        parsed_results = []
        all_lemmas = []

        progress_bar = st.progress(0, text="Consultando API do Ancient Greek WordNet...")
        words = [word for sentence in doc.sentences for word in sentence.words if word.upos != 'PUNCT']
        total_words = len(words)

        for idx, word in enumerate(words):
            lemma = word.lemma if word.lemma else word.text
            all_lemmas.append(lemma)
            
            synset_info = get_agwn_synsets(lemma)
            
            parsed_results.append({
                "Palavra": word.text,
                "Lema": lemma,
                "Classe (POS)": word.upos,
                "Synsets (AGWN)": synset_info
            })
            if total_words > 0:
                progress_bar.progress((idx + 1) / total_words)
        
        progress_bar.empty()
        df_agwn = pd.DataFrame(parsed_results)

        # B) ANÁLISE DE SENTIMENTO COM RÓTULOS MAPEADOS
        lemmatized_text = " ".join(all_lemmas)
        
        raw_output = sentiment_nlp(text_input)
        lemma_output = sentiment_nlp(lemmatized_text)

        raw_sentiment = raw_output[0] if isinstance(raw_output[0], list) else raw_output
        lemma_sentiment = lemma_output[0] if isinstance(lemma_output[0], list) else lemma_output

        # Aplicação da renomeação dos rótulos
        df_raw = pd.DataFrame(raw_sentiment)
        df_raw['label'] = df_raw['label'].map(lambda x: LABEL_MAP.get(x, x))
        if 'score' in df_raw.columns:
            df_raw['score'] = df_raw['score'].apply(lambda x: f"{x * 100:.2f}%")

        df_lem = pd.DataFrame(lemma_sentiment)
        df_lem['label'] = df_lem['label'].map(lambda x: LABEL_MAP.get(x, x))
        if 'score' in df_lem.columns:
            df_lem['score'] = df_lem['score'].apply(lambda x: f"{x * 100:.2f}%")

        # -----------------------------------------------------------------------------
        # 4. EXIBIÇÃO
        # -----------------------------------------------------------------------------
        tab1, tab2 = st.tabs(["📚 Synsets do AGWN", "💭 Análise de Sentimento"])

        with tab1:
            st.subheader("Lematização e Synsets")
            st.dataframe(df_agwn, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentimento: Texto Original")
                st.dataframe(df_raw, use_container_width=True)

            with col2:
                st.subheader("Sentimento: Texto Lematizado")
                st.dataframe(df_lem, use_container_width=True)

            st.subheader("Texto Lematizado Unificado")
            st.code(lemmatized_text, language="text")
