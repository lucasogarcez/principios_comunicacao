import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Gera um sinal binário
def gerar_sinal(n_bits, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n_bits)

# Adiciona ruído Gaussiano ao sinal
def adicionar_ruido(sinal, desvio_padrao):
    ruido = np.random.normal(0, desvio_padrao, len(sinal))
    return sinal + ruido

# Calcula BER
def calcular_ber(sinal_original, sinal_recebido):
    decodificado = np.where(sinal_recebido > 0.5, 1, 0)
    erros = np.sum(decodificado != sinal_original)
    return erros / len(sinal_original)

# Calcula SNR
def calcular_snr(sinal_original, sinal_recebido):
    pot_sinal = np.mean(sinal_original**2)
    pot_ruido = np.mean((sinal_recebido - sinal_original)**2)
    if pot_ruido == 0:
        return np.inf
    return 10 * np.log10(pot_sinal / pot_ruido)

# Interface do Streamlit
st.title("Simulação de Comunicação com Canal Ruidoso (BWSG)")

n_bits = st.slider("Número de bits", 10, 100, 50, step=10)
desvio_padrao = st.slider("Intensidade do Ruído (Desvio Padrão)", 0.0, 3.0, 0.5, step=0.1)

if 'seed' not in st.session_state:
    st.session_state.seed = np.random.randint(0, 10000)

sinal = gerar_sinal(n_bits, st.session_state.seed)
sinal_ruidoso = adicionar_ruido(sinal, desvio_padrao)

ber = calcular_ber(sinal, sinal_ruidoso)
snr = calcular_snr(sinal, sinal_ruidoso)

# Gráfico interativo
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(n_bits), y=sinal, mode='lines+markers', name='Sinal Original'))
fig.add_trace(go.Scatter(x=np.arange(n_bits), y=sinal_ruidoso, mode='lines+markers', name='Sinal Ruidoso'))
fig.update_layout(title='Transmissão com Ruído', xaxis_title='Bit', yaxis_title='Valor do Sinal')

st.plotly_chart(fig)

st.markdown(f"**Taxa de Erro de Bit (BER):** `{ber:.4f}`")
st.markdown(f"**SNR (dB):** `{snr:.2f}`")
