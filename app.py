import streamlit as st
import spacy
import pandas as pd
import requests
import urllib.parse
import unicodedata
from transformers import pipeline

st.set_page_config(
    page_title="Ancient Greek Engine: AGWN & Sentiment",
    page_icon="🏛️",
    layout="wide"
)

# Mapeamento de rótulos de sentimento
LABEL_MAP = {
    "LABEL_0": "LABEL_0 (Negativo)",
    "LABEL_1": "LABEL_1 (Positivo)",
    "LABEL_2": "LABEL_2 (Neutro)",
    "NEGATIVE": "NEGATIVE (Negativo)",
    "POSITIVE": "POSITIVE (Positivo)"
}

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO COM DOWNLOAD AUTOMÁTICO DO MODELO SPACY
# -----------------------------------------------------------------------------
@st.cache_resource
def load_spacy_greek():
    """Garante o download e carregamento do modelo de Grego Antigo no spaCy."""
    model_name = "grc_proiel_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        # Se o modelo não estiver instalado, baixa automaticamente via CLI do spaCy
        spacy.cli.download(model_name)
        return spacy.load(model_name)

@st.cache_resource
def load_sentiment_model():
    """Carrega o modelo de sentimento via pipeline da Hugging Face."""
    return pipeline(
        "sentiment-analysis",
        model="nlpaueb/bert-base-greek-uncased-v1",
        top_k=None
    )

with st.spinner("Instalando e carregando modelos de NLP em Grego Antigo..."):
    nlp = load_spacy_greek()
    sentiment_nlp = load_sentiment_model()

# -----------------------------------------------------------------------------
# 2. CONSULTA À API DO ANCIENT GREEK WORDNET
# -----------------------------------------------------------------------------
def remove_diacritics(text):
    """Remove acentos e espíritos do grego para busca de fallback."""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def query_agwn_endpoint(lemma_str):
    """Consulta à API oficial do AGWN com tratamento de URL."""
    encoded = urllib.parse.quote(lemma_str)
    url = f"https://greekwordnet.chs.harvard.edu/api/lemma/{encoded}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=3)
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
    lemma_clean = lemma.strip().lower()
    
    # 1ª tentativa com lema original
    res = query_agwn_endpoint(lemma_clean)
    if res:
        return res
    
    # 2ª tentativa sem acentos/diacríticos
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
        # A) PROCESSAMENTO MORFOLÓGICO VIA SPACY
        doc = nlp(text_input)
        parsed_results = []
        all_lemmas = []

        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        total_tokens = len(tokens)

        progress_bar = st.progress(0, text="Lematizando e consultando a API do AGWN...")

        for idx, token in enumerate(tokens):
            lemma = token.lemma_ if token.lemma_ else token.text
            all_lemmas.append(lemma)
            
            synset_info = get_agwn_synsets(lemma)
            
            parsed_results.append({
                "Palavra": token.text,
                "Lema": lemma,
                "Classe (POS)": token.pos_,
                "Synsets (AGWN)": synset_info
            })
            if total_tokens > 0:
                progress_bar.progress((idx + 1) / total_tokens)
        
        progress_bar.empty()
        df_agwn = pd.DataFrame(parsed_results)

        # B) ANÁLISE DE SENTIMENTO
        lemmatized_text = " ".join(all_lemmas)
        
        raw_output = sentiment_nlp(text_input)
        lemma_output = sentiment_nlp(lemmatized_text)

        raw_sentiment = raw_output[0] if isinstance(raw_output[0], list) else raw_output
        lemma_sentiment = lemma_output[0] if isinstance(lemma_output[0], list) else lemma_output

        df_raw = pd.DataFrame(raw_sentiment)
        df_raw['label'] = df_raw['label'].map(lambda x: LABEL_MAP.get(x, x))
        if 'score' in df_raw.columns:
            df_raw['score'] = df_raw['score'].apply(lambda x: f"{x * 100:.2f}%")

        df_lem = pd.DataFrame(lemma_sentiment)
        df_lem['label'] = df_lem['label'].map(lambda x: LABEL_MAP.get(x, x))
        if 'score' in df_lem.columns:
            df_lem['score'] = df_lem['score'].apply(lambda x: f"{x * 100:.2f}%")

        # -----------------------------------------------------------------------------
        # 4. EXIBIÇÃO DE RESULTADOS
        # -----------------------------------------------------------------------------
        tab1, tab2 = st.tabs(["📚 Synsets do AGWN", "💭 Análise de Sentimento"])

        with tab1:
            st.subheader("Lematização (spaCy) e Synsets (AGWN API)")
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