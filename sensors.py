import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cross_decomposition import PLSRegression

# Configuração da página para visualização industrial expandida
st.set_page_config(page_title="PIMS - Debutanizer Historian", layout="wide")

# Estilização CSS para aproximar a interface de um sistema supervisório (Dark Mode Industrial)
st.markdown("""
    <style>
    .stMetric {
        background-color: #1e2630;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #00d4ff;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #00ff66;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏭 Sistema Historiador de Processos (PIMS) - Refinaria")
st.subheader("Monitoramento Online de Variáveis Reais e Soft Sensor Adaptativo")

# ==============================================================================
# 1. Carregamento e Dicionário de Desnormalização
# ==============================================================================
@st.cache_data
def carregar_dados_brutos():
    url = "https://raw.githubusercontent.com/hkaneko1985/adaptive_soft_sensors/master/debutanizer_y_10.csv"
    data = pd.read_csv(url, index_col=0)
    data.columns = ['C4_Fundo', 'Vazao_Alim', 'Temp_Alim', 'Pressao_Topo',
                  'Pressao_Diff', 'Vazao_Refluxo', 'Temp_Refluxo', 'Temp_Bandeja']
    return data.ffill()

df_base = carregar_dados_brutos()

# Limites operacionais do benchmark de Fortuna para conversão de engenharia
limites_originais = {
    'Vazao_Alim':     {'min': 0.0, 'max': 70.0,  'unit': 'm³/h'},
    'Temp_Alim':      {'min': 0.0, 'max': 750.0, 'unit': '°C'},
    'Pressao_Topo':   {'min': 0.0, 'max': 15.0,  'unit': 'kg/cm²'},
    'Pressao_Diff':   {'min': 0.0, 'max': 1.0,   'unit': 'kg/cm²'},
    'Vazao_Refluxo':  {'min': 0.0, 'max': 350.0, 'unit': 'm³/h'},
    'Temp_Refluxo':   {'min': 0.0, 'max': 750.0, 'unit': '°C'},
    'Temp_Bandeja':   {'min': 0.0, 'max': 200.0, 'unit': '°C'},
    'C4_Fundo':       {'min': 0.0, 'max': 10.0,  'unit': '%'}
}

def desnormalizar(valor, variavel):
    v_min = limites_originais[variavel]['min']
    v_max = limites_originais[variavel]['max']
    return valor * (v_max - v_min) + v_min

# Parâmetros de sintonia do seu modelo MW-PLS
W_size = 400
n_comp = 6
delay_analisador = 22

# ==============================================================================
# 2. Painel de Controle Lateral (Painel de Operação)
# ==============================================================================
st.sidebar.header("Painel de Controle SCADA")
velocidade = st.sidebar.slider("Varredura do Historiador (Segundos)", 0.05, 1.5, 0.3)
executar_scada = st.sidebar.toggle("Coleta de Dados Ativa", value=True)

if 'passo_atual' not in st.session_state:
    st.session_state.passo_atual = W_size + delay_analisador + 1
# Históricos em unidades reais para plotagem industrial
if 'historico_tempo' not in st.session_state:
    st.session_state.historico_tempo = []
if 'hist_c4_real' not in st.session_state:
    st.session_state.hist_c4_real = []
if 'hist_c4_pred' not in st.session_state:
    st.session_state.hist_c4_pred = []
if 'hist_vazao_alim' not in st.session_state:
    st.session_state.hist_vazao_alim = []
if 'hist_temp_bandeja' not in st.session_state:
    st.session_state.hist_temp_bandeja = []

# ==============================================================================
# 3. Lógica de Varredura e Computação Adaptativa Local
# ==============================================================================
if executar_scada and st.session_state.passo_atual < len(df_base):
    t = st.session_state.passo_atual
    linha_norm = df_base.iloc[t]
    
    # Execução oculta do MW-PLS em escala [0, 1] para manter integridade numérica
    train_end = t - delay_analisador
    train_start = train_end - W_size
    janela_treino = df_base.iloc[train_start:train_end]
    
    X_train = janela_treino.drop(columns=['C4_Fundo']).values
    y_train = janela_treino['C4_Fundo'].values
    X_atual = linha_norm.drop('C4_Fundo').values.reshape(1, -1)
    
    std_window = np.std(X_train, axis=0)
    std_window[std_window < 1e-5] = 1.0
    mean_window = np.mean(X_train, axis=0)
    X_train_scaled = (X_train - mean_window) / std_window
    X_atual_scaled = (X_atual - mean_window) / std_window
    
    model = PLSRegression(n_components=n_comp)
    model.fit(X_train_scaled, y_train)
    pred_c4_norm = model.predict(X_atual_scaled).flatten()[0]
    
    # Conversão instantânea de escalas norm -> física real para exibição
    c4_real_fisico = desnormalizar(linha_norm['C4_Fundo'], 'C4_Fundo')
    c4_pred_fisico = desnormalizar(pred_c4_norm, 'C4_Fundo')
    vazao_alim_fisico = desnormalizar(linha_norm['Vazao_Alim'], 'Vazao_Alim')
    temp_alim_fisico = desnormalizar(linha_norm['Temp_Alim'], 'Temp_Alim')
    pres_topo_fisico = desnormalizar(linha_norm['Pressao_Topo'], 'Pressao_Topo')
    pres_diff_fisico = desnormalizar(linha_norm['Pressao_Diff'], 'Pressao_Diff')
    vazao_reflux_fisico = desnormalizar(linha_norm['Vazao_Refluxo'], 'Vazao_Refluxo')
    temp_reflux_fisico = desnormalizar(linha_norm['Temp_Refluxo'], 'Temp_Refluxo')
    temp_bandeja_fisico = desnormalizar(linha_norm['Temp_Bandeja'], 'Temp_Bandeja')
    
    # Armazenamento de dados convertidos (Janela máxima de 150 pontos em tela)
    st.session_state.historico_tempo.append(t)
    st.session_state.hist_c4_real.append(c4_real_fisico)
    st.session_state.hist_c4_pred.append(c4_pred_fisico)
    st.session_state.hist_vazao_alim.append(vazao_alim_fisico)
    st.session_state.hist_temp_bandeja.append(temp_bandeja_fisico)
    
    if len(st.session_state.historico_tempo) > 150:
        st.session_state.historico_tempo.pop(0)
        st.session_state.hist_c4_real.pop(0)
        st.session_state.hist_c4_pred.pop(0)
        st.session_state.hist_vazao_alim.pop(0)
        st.session_state.hist_temp_bandeja.pop(0)

    # ==============================================================================
    # 4. Arquitetura da Tela Visual (Estilo Console de Operador)
    # ==============================================================================
    col_dados, col_trends = st.columns([1, 2])
    
    with col_dados:
        st.markdown("### 🎛️ Leituras Atuais dos Instrumentos")
        
        st.markdown("**Seção de Alimentação da Planta**")
        st.metric(label="Vazão de Carga", value=f"{vazao_alim_fisico:.2f} m³/h")
        st.metric(label="Temperatura de Entrada", value=f"{temp_alim_fisico:.2f} °C")
        
        st.markdown("---")
        st.markdown("**Controle de Pressão e Refluxo (Topo)**")
        st.metric(label="Pressão Absoluta do Topo", value=f"{pres_topo_fisico:.2f} kg/cm²")
        st.metric(label="Pressão Diferencial (ΔP Interno)", value=f"{pres_diff_fisico:.3f} kg/cm²")
        st.metric(label="Vazão de Refluxo Condensado", value=f"{vazao_reflux_fisico:.2f} m³/h")
        st.metric(label="Temperatura do Refluxo", value=f"{temp_reflux_fisico:.2f} °C")
        
        st.markdown("---")
        st.markdown("**Zona de Retificação (Fundo)**")
        st.metric(label="Temperatura da 6ª Bandeja (T004)", value=f"{temp_bandeja_fisico:.2f} °C")

    with col_trends:
        st.markdown("### 📊 Gráficos de Tendências Históricas (Multitrend PIMS)")
        
        # Criação de múltiplos subplots com eixos Y independentes, típicos de softwares industriais
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Trend 1: Especificação de Qualidade Líquida (Teor de C4)", 
                                            "Trend 2: Hidrodinâmica da Alimentação", 
                                            "Trend 3: Perfil Térmico da Coluna"))
        
        # Subplot 1: Soft Sensor vs Cromatógrafo (Unidade Física Reais em %)
        fig.add_trace(go.Scatter(x=st.session_state.historico_tempo, y=st.session_state.hist_c4_real,
                                 mode='lines', name='Cromatógrafo (Lab %)', line=dict(color='#00ff66', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.historico_tempo, y=st.session_state.hist_c4_pred,
                                 mode='lines', name='Soft Sensor MW-PLS (%)', line=dict(color='#ff3333', width=2, dash='dot')), row=1, col=1)
        
        # Subplot 2: Vazão de Alimentação (m³/h)
        fig.add_trace(go.Scatter(x=st.session_state.historico_tempo, y=st.session_state.hist_vazao_alim,
                                 mode='lines', name='Carga (m³/h)', line=dict(color='#00d4ff', width=1.5)), row=2, col=1)
        
        # Subplot 3: Temperatura da Bandeja (°C)
        fig.add_trace(go.Scatter(x=st.session_state.historico_tempo, y=st.session_state.hist_temp_bandeja,
                                 mode='lines', name='TI-Bandeja (°C)', line=dict(color='#ffaa00', width=1.5)), row=3, col=1)
        
        # Customizações de Grid e Fundo Escuro para menor fadiga visual do operador
        fig.update_layout(height=650, template="plotly_dark", showlegend=True,
                          margin=dict(l=20, r=20, t=30, b=20),
                          legend=dict(orientation="h", y=1.06, x=0))
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2e3846')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2e3846')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Painel Dinâmico de Sistema Inter travamento/Alarmes Industriais
        st.markdown("#### 🔔 Matriz de Intertravamento e Alarmes")
        if c4_pred_fisico > 3.5:  # Limite crítico de engenharia química (ex: 3.5% de C4 no resíduo)
            st.markdown(f"<div style='background-color:#5a1818; padding:12px; border-radius:4px; border-left:6px solid #ff3333; color:white; font-weight:bold;'>"
                        f"⚠️ [ALERTA CRÍTICO HI-HI] Teor de Butano no fundo excedeu a janela de conformidade: {c4_pred_fisico:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#183618; padding:12px; border-radius:4px; border-left:6px solid #00ff66; color:white; font-weight:bold;'>"
                        f"✔️ [PROCESSO OK] Composição estimada dentro da especificação comercial: {c4_pred_fisico:.2f}%</div>", unsafe_allow_html=True)

    # Incremento do ponteiro temporal e reatualização automática da varredura
    st.session_state.passo_atual += 1
    time.sleep(velocidade)
    st.rerun()
