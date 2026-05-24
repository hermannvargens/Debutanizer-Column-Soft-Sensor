import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página para visualização industrial expandida
st.set_page_config(page_title="PIMS - Debutanizer Full Historian", layout="wide")

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

st.title(" PIMS - Historiador de Variáveis da Planta")
st.subheader("Varredura Completa dos Sensores da Coluna Debutanizadora em Unidades de Engenharia")

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

# Dicionário completo com os limites reais do benchmark industrial de Fortuna
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
st.sidebar.header("Configurações de Varredura")
velocidade = st.sidebar.slider("Taxa de Atualização (Segundos)", 0.05, 1.5, 0.2)
executar_scada = st.sidebar.toggle("Aquisição Ativa (Online)", value=True)

if 'passo_atual' not in st.session_state:
    st.session_state.passo_atual = 0

# Inicialização de buffers para todos os 8 sensores do dataset
if 'hist_tempo' not in st.session_state: st.session_state.hist_tempo = []
if 'hist_c4' not in st.session_state: st.session_state.hist_c4 = []
if 'hist_vazao_alim' not in st.session_state: st.session_state.hist_vazao_alim = []
if 'hist_temp_alim' not in st.session_state: st.session_state.hist_temp_alim = []
if 'hist_pres_topo' not in st.session_state: st.session_state.hist_pres_topo = []
if 'hist_pres_diff' not in st.session_state: st.session_state.hist_pres_diff = []
if 'hist_vazao_refluxo' not in st.session_state: st.session_state.hist_vazao_refluxo = []
if 'hist_temp_refluxo' not in st.session_state: st.session_state.hist_temp_refluxo = []
if 'hist_temp_bandeja' not in st.session_state: st.session_state.hist_temp_bandeja = []

# ==============================================================================
# 3. Laço de Captura e Conversão de Escala
# ==============================================================================
if executar_scada and st.session_state.passo_atual < len(df_base):
    t = st.session_state.passo_atual
    linha_norm = df_base.iloc[t]
    
    # Conversão de todas as variáveis do dataset para unidades de engenharia
    c4_f = desnormalizar(linha_norm['C4_Fundo'], 'C4_Fundo')
    vazao_a = desnormalizar(linha_norm['Vazao_Alim'], 'Vazao_Alim')
    temp_a = desnormalizar(linha_norm['Temp_Alim'], 'Temp_Alim')
    pres_t = desnormalizar(linha_norm['Pressao_Topo'], 'Pressao_Topo')
    pres_d = desnormalizar(linha_norm['Pressao_Diff'], 'Pressao_Diff')
    vazao_r = desnormalizar(linha_norm['Vazao_Refluxo'], 'Vazao_Refluxo')
    temp_r = desnormalizar(linha_norm['Temp_Refluxo'], 'Temp_Refluxo')
    temp_b = desnormalizar(linha_norm['Temp_Bandeja'], 'Temp_Bandeja')
    
    # Gravação nos buffers do Historiador
    st.session_state.hist_tempo.append(t)
    st.session_state.hist_c4.append(c4_f)
    st.session_state.hist_vazao_alim.append(vazao_a)
    st.session_state.hist_temp_alim.append(temp_a)
    st.session_state.hist_pres_topo.append(pres_t)
    st.session_state.hist_pres_diff.append(pres_d)
    st.session_state.hist_vazao_refluxo.append(vazao_r)
    st.session_state.hist_temp_refluxo.append(temp_r)
    st.session_state.hist_temp_bandeja.append(temp_b)
    
    # Janela deslizante de memória em tela (Mantém os últimos 150 pontos)
    if len(st.session_state.hist_tempo) > 150:
        for chave in ['hist_tempo', 'hist_c4', 'hist_vazao_alim', 'hist_temp_alim', 
                      'hist_pres_topo', 'hist_pres_diff', 'hist_vazao_refluxo', 
                      'hist_temp_refluxo', 'hist_temp_bandeja']:
            st.session_state[chave].pop(0)

    # ==============================================================================
    # 4. Painel de Exibição em Tela (Layout SCADA / PIMS)
    # ==============================================================================
    col_dados, col_trends = st.columns([1, 2.3])
    
    with col_dados:
        st.markdown("### Leituras Atuais")
        
        st.markdown("**Seção 100: Alimentação de Carga**")
        st.metric(label="FI-101 (Vazão de Carga)", value=f"{vazao_a:.2f} m³/h")
        st.metric(label="TI-101 (Temperatura de Carga)", value=f"{temp_a:.2f} °C")
        
        st.markdown("---")
        st.markdown("**Seção 200: Topo da Coluna**")
        st.metric(label="PI-201 (Pressão do Topo)", value=f"{pres_t:.2f} kg/cm²")
        st.metric(label="PDI-202 (Pressão Diferencial - ΔP)", value=f"{pres_d:.3f} kg/cm²")
        st.metric(label="FI-203 (Vazão de Refluxo)", value=f"{vazao_r:.2f} m³/h")
        st.metric(label="TI-203 (Temperatura do Refluxo)", value=f"{temp_r:.2f} °C")
        
        st.markdown("---")
        st.markdown("**Seção 300: Pratos de Controle e Fundo**")
        st.metric(label="TI-304 (Temperatura da 6ª Bandeja)", value=f"{temp_b:.2f} °C")
        st.metric(label="AI-301 (Teor de C4 no Resíduo - Cromatógrafo)", value=f"{c4_f:.3f} %")

    with col_trends:
        st.markdown("### Gráficos de Tendência Sincronizados (Multitrend)")
        
        # Criação de 4 subplots para acomodar todos os sensores sem poluição visual
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.06,
            subplot_titles=(
                "Trend 1: Qualidade do Fundo (Analisador)", 
                "Trend 2: Perfil Hidrodinâmico (Vazões)", 
                "Trend 3: Perfil de Pressão da Torre",
                "Trend 4: Perfil Térmico da Coluna"
            )
        )
        
        # Subplot 1: Concentração de C4 (% molar/massa)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_c4,
                                 mode='lines', name='AI-301 (C4 Fundo %)', line=dict(color='#00ff66', width=2.5)), row=1, col=1)
        
        # Subplot 2: Vazão de Carga vs Vazão de Refluxo (m³/h)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_vazao_alim,
                                 mode='lines', name='FI-101 (Carga m³/h)', line=dict(color='#00d4ff', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_vazao_refluxo,
                                 mode='lines', name='FI-203 (Refluxo m³/h)', line=dict(color='#3399ff', width=1.5, dash='dash')), row=2, col=1)
        
        # Subplot 3: Pressão de Topo vs Pressão Diferencial (kg/cm²)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_pres_topo,
                                 mode='lines', name='PI-201 (Topo kg/cm²)', line=dict(color='#ffcc00', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_pres_diff,
                                 mode='lines', name='PDI-202 (ΔP kg/cm²)', line=dict(color='#e0e0e0', width=1.2, dash='dot')), row=3, col=1)
        
        # Subplot 4: Temperaturas da Carga, Refluxo e 6ª Bandeja (°C)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_temp_alim,
                                 mode='lines', name='TI-101 (Carga °C)', line=dict(color='#ff6600', width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_temp_refluxo,
                                 mode='lines', name='TI-203 (Refluxo °C)', line=dict(color='#ff9966', width=1.2, dash='dash')), row=4, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.hist_tempo, y=st.session_state.hist_temp_bandeja,
                                 mode='lines', name='TI-304 (Bandeja 6 °C)', line=dict(color='#ff00ff', width=1.5)), row=4, col=1)
        
        # Correção final do layout para afastar as legendas do título do Trend 1
        fig.update_layout(
            height=750, 
            template="plotly_dark", 
            showlegend=True,
            margin=dict(l=20, r=20, t=80, b=20),
            legend=dict(
                orientation="h", 
                y=1.07, 
                x=0,
                xanchor="left",
                yanchor="bottom"
            )
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#252e38')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#252e38')
        
        st.plotly_chart(fig, use_container_width=True)

    # Avanço do ponteiro lógico do historiador
    st.session_state.passo_atual += 1
    time.sleep(velocidade)
    st.rerun()
