import streamlit as st
import stanza
import pandas as pd
import requests
import urllib.parse
from transformers import pipeline

st.set_page_config(
    page_title="Ancient Greek Engine: AGWN & Sentiment",
    page_icon="🏛️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO E CACHE DOS MODELOS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_stanza():
    stanza.download('grc', verbose=False)
    return stanza.Pipeline('grc', processors='tokenize,mwt,pos,lemma', verbose=False)

@st.cache_resource
def load_sentiment_model():
    # Carrega pipeline de análise de sentimento
    return pipeline(
        "sentiment-analysis",
        model="nlpaueb/bert-base-greek-uncased-v1",
        top_k=None # Retorna todos os rótulos de pontuação
    )

with st.spinner("Carregando modelos de NLP em Grego Antigo..."):
    stanza_nlp = load_stanza()
    sentiment_nlp = load_sentiment_model()

# -----------------------------------------------------------------------------
# 2. CONSULTA À API DO ANCIENT GREEK WORDNET / LOGEION
# -----------------------------------------------------------------------------
def get_agwn_synsets(lemma):
    """Consulta os synsets/definições lexicográficas tratando diacríticos e conexões."""
    encoded_lemma = urllib.parse.quote(lemma)
    url = f"https://greekwordnet.chs.harvard.edu/api/lemma/{encoded_lemma}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StreamlitApp/1.0"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code == 200:
            data = response.json()
            synsets_found = []
            
            items = data if isinstance(data, list) else data.get("synsets", [])
            for item in items:
                synset_id = item.get("synset_id", item.get("id", "N/A"))
                gloss = item.get("gloss", item.get("definition", "Sem definição disponível"))
                pos = item.get("pos", "")
                synsets_found.append(f"**[{synset_id}]** ({pos}) {gloss}")
            
            return " | ".join(synsets_found) if synsets_found else "Nenhum synset retornado"
        else:
            return f"Não localizado na base (HTTP {response.status_code})"
    except Exception as e:
        return f"Sem resposta do servidor AGWN"

# -----------------------------------------------------------------------------
# 3. INTERFACE PRINCIPAL
# -----------------------------------------------------------------------------
st.title("🏛️ Ancient Greek Engine: AGWN API & Sentiment")

default_text = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"
text_input = st.text_area("Insira o texto em Grego Antigo:", value=default_text, height=120)

if st.button("🚀 Processar com AGWN & Hugging Face", type="primary"):
    if not text_input.strip():
        st.warning("Por favor, insira um texto em Grego Antigo.")
    else:
        # A) PROCESSAMENTO MORFOLÓGICO & AGWN
        doc = stanza_nlp(text_input)
        parsed_results = []
        all_lemmas = []

        progress_bar = st.progress(0, text="Consultando a API do Ancient Greek WordNet...")
        
        words = [word for sentence in doc.sentences for word in sentence.words]
        total_words = len(words)

        for idx, word in enumerate(words):
            lemma = word.lemma if word.lemma else word.text
            all_lemmas.append(lemma)
            
            agwn_data = get_agwn_synsets(lemma)
            
            parsed_results.append({
                "Palavra": word.text,
                "Lema": lemma,
                "Classe (POS)": word.upos,
                "Synsets (AGWN API)": agwn_data
            })
            
            progress_bar.progress((idx + 1) / total_words)
        
        progress_bar.empty()
        df_agwn = pd.DataFrame(parsed_results)

        # B) ANÁLISE DE SENTIMENTO VIA TRANSFORMER (Tratamento Seguro)
        lemmatized_text = " ".join(all_lemmas)
        
        raw_output = sentiment_nlp(text_input)
        lemma_output = sentiment_nlp(lemmatized_text)

        # Normalização dos dados para DataFrame do Pandas
        raw_sentiment = raw_output[0] if isinstance(raw_output[0], list) else raw_output
        lemma_sentiment = lemma_output[0] if isinstance(lemma_output[0], list) else lemma_output

        # -----------------------------------------------------------------------------
        # 4. EXIBIÇÃO DE RESULTADOS
        # -----------------------------------------------------------------------------
        tab1, tab2 = st.tabs(["📚 Synsets do AGWN", "💭 Análise de Sentimento (Hugging Face)"])

        with tab1:
            st.subheader("Lematização e Consulta à API do Ancient Greek WordNet")
            st.dataframe(df_agwn, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentimento: Texto Original")
                df_raw = pd.DataFrame(raw_sentiment)
                if 'score' in df_raw.columns:
                    df_raw['score'] = df_raw['score'].apply(lambda x: f"{x * 100:.2f}%")
                st.dataframe(df_raw, use_container_width=True)

            with col2:
                st.subheader("Sentimento: Texto Lematizado")
                df_lem = pd.DataFrame(lemma_sentiment)
                if 'score' in df_lem.columns:
                    df_lem['score'] = df_lem['score'].apply(lambda x: f"{x * 100:.2f}%")
                st.dataframe(df_lem, use_container_width=True)

            st.subheader("Texto Lematizado Unificado")
            st.code(lemmatized_text, language="text")