import streamlit as st
import stanza
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

# Configuração inicial da página
st.set_page_config(
    page_title="Ancient Greek WordNet & Sentiment",
    page_icon="🏛️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. DOWNLOAD E CARREGAMENTO DE MODELOS (COM CACHE)
# -----------------------------------------------------------------------------
@st.cache_resource
def setup_nlp():
    """Baixa e carrega todos os recursos necessários do NLTK e Stanza."""
    # Downloads explícitos do NLTK para WordNet Multilíngue (OMW)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('omw', quiet=True)
    nltk.download('extended_omw', quiet=True)
    
    # Download e inicialização do pipeline do Stanza para Grego Antigo
    stanza.download('grc', verbose=False)
    
    return stanza.Pipeline('grc', processors='tokenize,mwt,pos,lemma', verbose=False)

# Inicialização global do pipeline
with st.spinner("Carregando modelos de NLP em Grego Antigo e WordNet..."):
    nlp = setup_nlp()

# -----------------------------------------------------------------------------
# 2. INTERFACE PRINCIPAL
# -----------------------------------------------------------------------------
st.title("🏛️ Ancient Greek NLP: Synsets & Sentiment App")
st.markdown("""
Esta aplicação realiza a **lematização de textos em Grego Antigo**, mapeia os **synsets** no WordNet 
e organiza o vocabulário para análises semânticas.
""")

st.sidebar.header("Configurações")
max_synsets = st.sidebar.slider("Máx. de Synsets por Lema:", 1, 5, 2)

default_text = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"
text_input = st.text_area("Insira o texto em Grego Antigo:", value=default_text, height=120)

# -----------------------------------------------------------------------------
# 3. PROCESSAMENTO DO TEXTO
# -----------------------------------------------------------------------------
if st.button("🚀 Processar Texto", type="primary"):
    if not text_input.strip():
        st.warning("Por favor, insira um texto para processar.")
    else:
        doc = nlp(text_input)
        
        parsed_data = []
        all_lemmas = []

        for sentence in doc.sentences:
            for word in sentence.words:
                lemma = word.lemma if word.lemma else word.text
                pos = word.upos
                all_lemmas.append(lemma)
                
                # Busca de Synsets com tratamento seguro de erros
                synsets = []
                try:
                    # Tenta buscar usando o código do idioma grego antigo / moderno no OMW
                    synsets = wn.synsets(lemma, lang='grc')
                except Exception:
                    try:
                        synsets = wn.synsets(lemma, lang='ell')
                    except Exception:
                        synsets = []

                synset_info = []
                if synsets:
                    for s in synsets[:max_synsets]:
                        defn = s.definition()
                        synset_info.append(f"**{s.name()}**: _{defn}_")
                
                parsed_data.append({
                    "Palavra Original": word.text,
                    "Lema": lemma,
                    "Classe Gramatical (POS)": pos,
                    "Synsets Encontrados": " | ".join(synset_info) if synset_info else "Nenhum synset direto"
                })

        df_results = pd.DataFrame(parsed_data)

        # -------------------------------------------------------------------------
        # 4. EXIBIÇÃO DOS RESULTADOS
        # -------------------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["📊 Tabela de Synsets", "🔍 Análise Léxica", "💭 Sentimento / Contexto"])

        with tab1:
            st.subheader("Lematização e Mapeamento de Synsets")
            st.dataframe(df_results, use_container_width=True)

        with tab2:
            st.subheader("Distribuição de Lemas do Texto")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total de Tokens", len(df_results))
                st.metric("Total de Lemas Únicos", len(set(all_lemmas)))
            
            with col2:
                st.write("**Lista de Lemas extraídos:**")
                st.write(", ".join(f"`{l}`" for l in set(all_lemmas)))

        with tab3:
            st.subheader("Análise Semântica e Polaridade")
            st.markdown("""
            > **Nota sobre Sentimento em Grego Antigo:**  
            > Como os lemas já foram isolados, você pode correlacionar esta lista com um léxico de polaridade 
            > ou passar os lemas para modelos de Embeddings.
            """)
            st.write("**Visualização do Texto Lematizado:**")
            st.code(" ".join(all_lemmas), language="text")