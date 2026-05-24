import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página para visualização industrial expandida
st.set_page_config(page_title="PIMS - Debutanizer Historian", layout="wide")

# Estilização CSS para o Modo Escuro de Sala de Controle (Interface SCADA)
st.markdown("""
    <style>
    .stMetric {
        background-color: #161b22;
        padding: 12px;
        border-radius: 4px;
        border-left: 5px solid #00d4ff;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #00ff66;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏭 Sistema Historiador de Processos (PIMS) - Console de Operação")
st.subheader("Varredura de Variáveis Brutas da Coluna Debutanizadora em Unidades de Engenharia")

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

# Dicionário de limites baseado no benchmark industrial de Fortuna
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

# ==============================================================================
# 2. Painel Lateral (Configuração de Escaneamento SCADA)
# ==============================================================================
st.sidebar.header("Configurações do SCADA")
velocidade = st.sidebar.slider("Taxa de Varredura (Segundos)", 0.05, 1.5, 0.2)
executar_scada = st.sidebar.toggle("Aquisição de Dados Ativa", value=True)

# Definição do ponto de partida da simulação
if 'passo_atual' not in st.session_state:
    st.session_state.passo_atual = 0

# Inicialização dos buffers de memória do historiador (máximo de 200 pontos em tela)
if 'hist_tempo' not in st.session_state: st.session_state.hist_tempo = []
if 'hist_c4' not in st.session_state: st.session_state.hist_c4 = []
if 'hist_vazao_alim' not in st.session_state: st.session_state.hist_vazao_alim = []
  if 'hist_temp_alim' not in st.session_state: st.session_state.hist_temp_alim = []
if 'hist_pres_topo' not in st.session_state: st.session_state.hist_pres_topo = []
if 'hist_vazao_refluxo' not in st.session_state: st.session_state.hist_vazao_refluxo = []
if 'hist_temp_bandeja' not in st.session_state: st.session_state.hist_temp_bandeja = []

# ==============================================================================
# 3. Laço de Captura e Conversão de Escala
# ==============================================================================
if executar_scada and st.session_state.passo_atual < len(df_base):
    t = st.session_state.passo_atual
    linha_norm = df_base.iloc[t]
    
    # Conversão instantânea da matriz normalizada para unidades físicas reais
    c4_fisico = desnormalizar(linha_norm['C4_Fundo'], 'C4_Fundo')
    vazao_alim_fisico = desnormalizar(linha_norm['Vazao_Alim'], 'Vazao_Alim')
    temp_alim_fisico = desnormalizar(linha_norm['Temp_Alim'], 'Temp_Alim')
    pres_topo_fisico = desnormalizar(linha_norm['Pressao_Topo'], 'Pressao_Topo')
    pres_diff_fisico = desnormalizar(linha_norm['Pressao_Diff'], 'Pressao_Diff')
    vazao_reflux_fisico = desnormalizar(linha_norm['Vazao_Refluxo'], 'Vazao_Refluxo')
    temp_reflux_fisico = desnormalizar(linha_norm['Temp_Refluxo'], 'Temp_Refluxo')
    temp_bandeja_fisico = desnormalizar(linha_norm['Temp_Bandeja'], 'Temp_Bandeja')
    
    # Alimentação dos buffers do historiador
    st.session_state.hist_tempo.append(t)
    st.session_state.hist_c4.append(c4_fisico)
    st.session_state.hist_vazao_alim.append(vazao_alim_fisico)
    st.session_state.hist_temp_alim.append(temp_alim_fisico)
    st.session_state.hist_pres_topo.append(pres_topo_fisico)
    st.session_state.hist_vazao_refluxo.append(vazao_reflux_fisico)
    st.session_state.hist_temp_bandeja.append(temp_bandeja_fisico)
    
    # Limitador de janela deslizante para manter a fluidez da tela
    if len(st.session_state.hist_tempo) > 200:
        st.session_state.hist_tempo.pop(0)
        st.session_state.hist_c4.pop(0)
        st.session_state.hist_vazao_alim.pop(0)
        st.session_state.hist_temp_alim.pop(0)
        st.session_state.hist_pres_topo.pop(0)
        st.session_state.hist_vazao_refluxo.pop(0)
        st.session_state.hist_temp_bandeja.pop(0)

    # ==============================================================================
    # 4. Organização do Dashboard (Layout SCADA / PIMS)
    # ==============================================================================
    col_dados, col_trends = st.columns([1, 2.5])
    
    with col_dados:
        st.markdown("### 🎛️ Painel de Medições Atuais")
        
        st.markdown("**Subsistema 100: Alimentação da Coluna**")
        st.metric(label="FI-101 (Vazão de Carga)", value=f"{vazao_alim_fisico:.2f} m³/h")
        st.metric(label="TI-101 (Temperatura da Carga)", value=f"{temp_alim_fisico:.2f} °C")
        
        st.markdown("---")
        st.markdown("**Subsistema 200: Controle do Topo**")
        st.metric(label="PI-201 (Pressão do Topo)", value=f"{pres_topo_fisico:.2f} kg/cm²")
        st.metric(label="PDI-202 (Pressão Diferencial Interna)", value=f"{pres_diff_fisico:.3f} kg/cm²")
        st.metric(label="FI-203 (Vazão de Refluxo)", value=f"{vazao_reflux_fisico:.2f} m³/h")
        st.metric(label="TI-203 (Temperatura do Refluxo)", value=f"{temp_reflux_fisico:.2f} °C")
        
        st.markdown("---")
        st.markdown("**Subsistema 300: Seção de Fundo**")
        st.metric(label="TI-304 (Temperatura da 6ª Bandeja)", value=f"{temp_bandeja_fisico:.2f} °C")
        st.metric(label="AI-301 (Teor de C4 no Resíduo - Lab)", value=f"{c4_fisico:.3f} %")

    with col_trends:
        st.markdown("### 📊 Multitrend Histórico (Eixos Sincronizados)")
        
        # Geração de gráficos empilhados para análise de transientes industriais
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Trend 1: Cromatógrafo de Fundo (% C4)", 
                                            "Trend 2: Balanço de Massa na Carga (m³/h)", 
                                            "Trend 3: Pressão de Topo (kg/cm²)",
                                            "Trend 4: Perfil Térmico (Carga vs Bandeja 6)"))
        
        # Subplot 1: Concentração Real de C4 (%)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_c4,
                                 mode='lines', name='AI-301 (% C4)', line=dict(color='#00ff66', width=2)), row=1, col=1)
        
        # Subplot 2: Vazão de Alimentação (m³/h)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_vazao_alim,
                                 mode='lines', name='FI-101 (m³/h)', line=dict(color='#00d4ff', width=1.5)), row=2, col=1)
        
        # Subplot 3: Pressão de Topo (kg/cm²)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_pres_topo,
                                 mode='lines', name='PI-201 (kg/cm²)', line=dict(color='#ffcc00', width=1.5)), row=3, col=1)
        
        # Subplot 4: Temperaturas da Carga e da 6ª Bandeja (°C)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_temp_alim,
                                 mode='lines', name='TI-101 Carga (°C)', line=dict(color='#ff6600', width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_temp_bandeja,
                                 mode='lines', name='TI-304 Bandeja (°C)', line=dict(color='#ff00ff', width=1.5)), row=4, col=1)
        
        # Configuração do tema escuro padrão para sistemas industriais
        fig.update_layout(height=700, template="plotly_dark", showlegend=True,
                          margin=dict(l=20, r=20, t=30, b=20),
                          legend=dict(orientation="h", y=1.05, x=0))
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#252e38')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#252e38')
        
        st.plotly_chart(fig, use_container_width=True)

    # Avança o ponteiro de linha do banco de dados e atualiza a varredura
    st.session_state.passo_atual += 1
    time.sleep(velocidade)
    st.rerun()
