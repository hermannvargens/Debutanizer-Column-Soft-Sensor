import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression

# Configuração da página do Streamlit
st.set_page_config(page_title="Dashboard Debutanizer Real-Time", layout="wide")

st.title("🏭 Monitoramento em Tempo Real - Coluna Debutanizadora")
st.subheader("Soft Sensor Adaptativo via Moving Window PLS (MW-PLS)")

# ==============================================================================
# 1. Carregamento dos Dados para a Simulação Online
# ==============================================================================
@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/hkaneko1985/adaptive_soft_sensors/master/debutanizer_y_10.csv"
    data = pd.read_csv(url, index_col=0)
    data.columns = ['C4_Fundo', 'Vazao_Alim', 'Temp_Alim', 'Pressao_Topo',
                  'Pressao_Diff', 'Vazao_Refluxo', 'Temp_Refluxo', 'Temp_Bandeja']
    return data.ffill()

df_base = carregar_dados()

# Variáveis do seu Modelo MW-PLS Otimizado
W_size = 400
n_comp = 6
delay_analisador = 22

# ==============================================================================
# 2. Layout da Interface (Colunas)
# ==============================================================================
col_pfd, col_graficos = st.columns([1, 1.2])

# Sidebar para Controle da Simulação
st.sidebar.header("Controle de Operação")
velocidade = st.sidebar.slider("Velocidade de Atualização (s)", 0.1, 2.0, 0.5)
loop_simulacao = st.sidebar.toggle("Iniciar Fluxo de Dados", value=True)

# Inicializadores de Estado do Streamlit (Session State)
if 'passo_atual' not in st.session_state:
    st.session_state.passo_atual = W_size + delay_analisador + 1
if 'historico_real' not in st.session_state:
    st.session_state.historico_real = []
if 'historico_pred' not in st.session_state:
    st.session_state.historico_pred = []

# ==============================================================================
# 3. Execução do Loop de Tempo Real
# ==============================================================================
if loop_simulacao:
    t = st.session_state.passo_atual
    
    if t < len(df_base):
        # Captura a linha atual do SCADA
        linha_atual = df_base.iloc[t]
        
        # --- Cálculo Online do MW-PLS (Seu algoritmo local) ---
        train_end = t - delay_analisador
        train_start = train_end - W_size
        
        janela_treino = df_base.iloc[train_start:train_end]
        X_train = janela_treino.drop(columns=['C4_Fundo']).values
        y_train = janela_treino['C4_Fundo'].values
        X_atual = linha_atual.drop('C4_Fundo').values.reshape(1, -1)
        
        # Escalonamento Local Protetor
        std_window = np.std(X_train, axis=0)
        std_window[std_window < 1e-5] = 1.0
        mean_window = np.mean(X_train, axis=0)
        X_train_scaled = (X_train - mean_window) / std_window
        X_atual_scaled = (X_atual - mean_window) / std_window
        
        # Ajuste do Modelo e Predição Instantânea
        model = PLSRegression(n_components=n_comp)
        model.fit(X_train_scaled, y_train)
        predicao_c4 = model.predict(X_atual_scaled).flatten()[0]
        
        # Salva históricos para os gráficos
        st.session_state.historico_real.append(linha_atual['C4_Fundo'])
        st.session_state.historico_pred.append(predicao_c4)
        
        # Limitador de histórico em tela para não sobrecarregar a memória
        if len(st.session_state.historico_real) > 100:
            st.session_state.historico_real.pop(0)
            st.session_state.historico_pred.pop(0)

        # ----------------------------------------------------------------------
        # COLUNA 1: Ilustração dos Sensores (Esquema Físico)
        # ----------------------------------------------------------------------
        with col_pfd:
            st.write("### 📐 Estado Físico dos Sensores")
            
            # Cards de Indicadores Industriais (Simulando o painel de controle)
            st.metric(label="💧 Vazão de Alimentação", value=f"{linha_atual['Vazao_Alim']:.4f}")
            st.metric(label="🔥 Temperatura da Carga", value=f"{linha_atual['Temp_Alim']:.4f}")
            
            st.markdown("---")
            st.markdown("**Variáveis Operacionais da Coluna:**")
            
            c1, c2 = st.columns(2)
            c1.metric(label="🔝 Pressão de Topo", value=f"{linha_atual['Pressao_Topo']:.4f}")
            c2.metric(label="📉 Pressão Diferencial ($\Delta P$)", value=f"{linha_atual['Pressao_Diff']:.4f}")
            
            c3, c4 = st.columns(2)
            c3.metric(label="🔄 Vazão de Refluxo", value=f"{linha_atual['Vazao_Refluxo']:.4f}")
            c4.metric(label="🌡️ Temperatura da 6ª Bandeja", value=f"{linha_atual['Temp_Bandeja']:.4f}")

        # ----------------------------------------------------------------------
        # COLUNA 2: Gráfico Dinâmico de Tendência
        # ----------------------------------------------------------------------
        with col_graficos:
            st.write("### 📈 Predição de Qualidade (Teor de C4 no Fundo)")
            
            # Montagem do Gráfico Interativo com Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.historico_real, mode='lines', 
                                     name='Cromatógrafo (Real)', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(y=st.session_state.historico_pred, mode='lines', 
                                     name='Soft Sensor (MW-PLS)', line=dict(color='crimson', dash='dash')))
            
            fig.update_layout(xaxis_title="Tempo Operacional (Amostras)", yaxis_title="Teor de C4 Normalizado",
                              margin=dict(l=20, r=20, t=20, b=20), height=400, legend=dict(orientation="h", y=1.1))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # KPI de Alerta de Qualidade
            if predicao_c4 > 0.4:
                st.error(f"🚨 ALERTA: Teor de C4 acima do limite especificado! Valor: {predicao_c4:.4f}")
            else:
                st.success(f"✅ Operação Estável: Teor de C4 sob controle. Valor: {predicao_c4:.4f}")

        # Avança o ponteiro da amostragem e força o rerun da interface
        st.session_state.passo_atual += 1
        time.sleep(velocidade)
        st.rerun()
