import streamlit as st
import stanza
import pandas as pd
import requests
from transformers import pipeline

st.set_page_config(
    page_title="Ancient Greek Engine: WordNet API & Sentiment",
    page_icon="🏛️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO E CACHE DOS MODELOS (Stanza + Transformer)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_stanza():
    """Carrega o pipeline do Stanza para Grego Antigo."""
    stanza.download('grc', verbose=False)
    return stanza.Pipeline('grc', processors='tokenize,mwt,pos,lemma', verbose=False)

@st.cache_resource
def load_sentiment_model():
    """Carrega o modelo Transformer da Hugging Face para análise de sentimento."""
    return pipeline(
        "sentiment-analysis",
        model="nlpaueb/bert-base-greek-uncased-v1",
        return_all_scores=True
    )

with st.spinner("Inicializando Stanza e modelo de linguagem Hugging Face..."):
    stanza_nlp = load_stanza()
    sentiment_nlp = load_sentiment_model()

# -----------------------------------------------------------------------------
# 2. FUNÇÃO DE CONSULTA À API DO ANCIENT GREEK WORDNET (HARVARD CHS)
# -----------------------------------------------------------------------------
def get_agwn_synsets(lemma):
    """Consulta a API oficial do Ancient Greek WordNet em Harvard."""
    url = f"https://greekwordnet.chs.harvard.edu/api/lemma/{lemma}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            synsets_found = []
            
            # Trata respostas em lista ou dicionário da API
            items = data if isinstance(data, list) else data.get("synsets", [])
            for item in items:
                synset_id = item.get("synset_id", item.get("id", "N/A"))
                gloss = item.get("gloss", item.get("definition", "Sem definição disponível"))
                pos = item.get("pos", "")
                synsets_found.append(f"**[{synset_id}]** ({pos}) {gloss}")
            
            return " | ".join(synsets_found) if synsets_found else "Nenhum synset na API"
        else:
            return f"Não encontrado (HTTP {response.status_code})"
    except Exception:
        return "Erro de conexão com a API do AGWN"

# -----------------------------------------------------------------------------
# 3. INTERFACE PRINCIPAL
# -----------------------------------------------------------------------------
st.title("🏛️ Ancient Greek Engine: AGWN API & Sentiment")
st.markdown("""
Esta aplicação realiza a **lematização**, consulta a **API oficial do Ancient Greek WordNet (Harvard CHS)** 
e calcula a **polaridade de sentimento contextual** através de modelos Transformer.
""")

default_text = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"
text_input = st.text_area("Insira o texto em Grego Antigo:", value=default_text, height=120)

if st.button("🚀 Processar com AGWN & Hugging Face", type="primary"):
    if not text_input.strip():
        st.warning("Por favor, insira um texto em Grego Antigo.")
    else:
        # A) PROCESSAMENTO MORFOLÓGICO & CONSULTA AGWN
        doc = stanza_nlp(text_input)
        parsed_results = []
        all_lemmas = []

        progress_bar = st.progress(0, text="Consultando a API do Ancient Greek WordNet...")
        
        words = [word for sentence in doc.sentences for word in sentence.words]
        total_words = len(words)

        for idx, word in enumerate(words):
            lemma = word.lemma if word.lemma else word.text
            all_lemmas.append(lemma)
            
            # Chamada à API do Harvard AGWN pelo lema extraído
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

        # B) ANÁLISE DE SENTIMENTO VIA TRANSFORMER (Contextual)
        lemmatized_text = " ".join(all_lemmas)
        
        # Sentimento da frase original
        raw_sentiment = sentiment_nlp(text_input)[0]
        # Sentimento da frase lematizada
        lemma_sentiment = sentiment_nlp(lemmatized_text)[0]

        # -----------------------------------------------------------------------------
        # 4. EXIBIÇÃO DOS RESULTADOS
        # -----------------------------------------------------------------------------
        tab1, tab2 = st.tabs(["📚 Synsets do AGWN (Harvard)", "💭 Análise de Sentimento (Hugging Face)"])

        with tab1:
            st.subheader("Lematização e Consulta à API do Ancient Greek WordNet")
            st.dataframe(df_agwn, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentimento: Texto Original (Com Flexão)")
                df_raw_sent = pd.DataFrame(raw_sentiment)
                df_raw_sent['score'] = df_raw_sent['score'].apply(lambda x: f"{x * 100:.2f}%")
                st.dataframe(df_raw_sent, use_container_width=True)
                
                top_raw = max(raw_sentiment, key=lambda x: x['score'])
                st.info(f"**Classe Predominante:** {top_raw['label']} ({top_raw['score']*100:.1f}%)")

            with col2:
                st.subheader("Sentimento: Texto Lematizado")
                df_lemma_sent = pd.DataFrame(lemma_sentiment)
                df_lemma_sent['score'] = df_lemma_sent['score'].apply(lambda x: f"{x * 100:.2f}%")
                st.dataframe(df_lemma_sent, use_container_width=True)
                
                top_lemma = max(lemma_sentiment, key=lambda x: x['score'])
                st.info(f"**Classe Predominante:** {top_lemma['label']} ({top_lemma['score']*100:.1f}%)")

            st.subheader("Texto Lematizado Unificado")
            st.code(lemmatized_text, language="text")