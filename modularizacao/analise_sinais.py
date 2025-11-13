import numpy as np
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Frequências dos sinais
F_M1 = 1000.0  # Hz (para vm1)
F_M2 = 2000.0  # Hz (para vm2)
F_M3_A = 1000.0 # Hz (para vm3)
F_M3_B = 3000.0 # Hz (para vm3)
F_C = 10000.0  # Hz (portadora)

# Parâmetros de amostragem
# A frequência modulada mais alta será Fc + (F_M3_A + F_M3_B) = 10k + 4k = 14k
# Precisamos de Fs > 2 * 14k = 28k. Vamos usar 200k para uma boa resolução.
F_S = 40e3    # Taxa de amostragem (Hz)
N = 2**17      # Número de pontos (potência de 2 para FFT eficiente)
T_MAX = N / F_S  # Tempo total de simulação (s)
t = np.linspace(0, T_MAX, N, endpoint=False) # Vetor de tempo

# Vetor de frequência para FFT
# Vamos mostrar apenas as frequências positivas (metade do vetor)
f = fftfreq(N, 1/F_S)[:N//2]

# Tempo mais curto para visualização dos gráficos (ex: 5ms)
T_PLOT_MAX = 0.005 # s
N_PLOT = int(T_PLOT_MAX * F_S)
t_plot = t[:N_PLOT]

# --- FUNÇÃO AUXILIAR PARA ESPECTRO ---

def get_spectrum(signal):
    """Calcula a magnitude do espectro (FFT) para um sinal."""
    # Normaliza por N e multiplica por 2 para obter a amplitude correta
    # (exceto DC)
    yf = 2.0/N * np.abs(fft(signal)[:N//2])
    return yf

# --- DEFINIÇÃO DOS SINAIS ---

# Sinais de Mensagem (vm)
vm1 = np.cos(2 * np.pi * F_M1 * t)
vm2 = 2 * np.cos(2 * np.pi * F_M1 * t) + np.cos(2 * np.pi * F_M2 * t)
vm3 = np.cos(2 * np.pi * F_M3_A * t) * np.cos(2 * np.pi * F_M3_B * t)
# Pela identidade produto-soma: vm3 = 0.5*cos(2pi*2000t) + 0.5*cos(2pi*4000t)

# Sinal da Portadora (vc)
vc = np.cos(2 * np.pi * F_C * t)

# Sinais Modulados (v_mod)
# Usaremos Modulação AM-LC: v_mod = (Ac + vm(t)) * cos(2*pi*Fc*t)
# Escolhemos Ac para evitar sobremodulação (Ac > max(|vm(t)|))

# Caso 1: max(|vm1|) = 1. Usaremos Ac = 1.5
A_c1 = 1.5
vmod1 = (A_c1 + vm1) * np.cos(2 * np.pi * F_C * t)
env1_pos = A_c1 + vm1
env1_neg = -(A_c1 + vm1)

# Caso 2: max(|vm2|) = 3. Usaremos Ac = 3.5
A_c2 = 3.5
vmod2 = (A_c2 + vm2) * np.cos(2 * np.pi * F_C * t)
env2_pos = A_c2 + vm2
env2_neg = -(A_c2 + vm2)

# Caso 3: max(|vm3|) = 1. Usaremos Ac = 1.5
A_c3 = 1.5
vmod3 = (A_c3 + vm3) * np.cos(2 * np.pi * F_C * t)
env3_pos = A_c3 + vm3
env3_neg = -(A_c3 + vm3)


# --- CÁLCULO DOS ESPECTROS ---

spec_vm1 = get_spectrum(vm1)
spec_vm2 = get_spectrum(vm2)
spec_vm3 = get_spectrum(vm3)
spec_vc = get_spectrum(vc)
spec_vmod1 = get_spectrum(vmod1)
spec_vmod2 = get_spectrum(vmod2)
spec_vmod3 = get_spectrum(vmod3)

# --- PLOTAGEM ---

# Figura 1: Sinais de Mensagem
fig1 = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "v_m1(t) - Tempo", "V_m1(f) - Frequência",
        "v_m2(t) - Tempo", "V_m2(f) - Frequência",
        "v_m3(t) - Tempo", "V_m3(f) - Frequência"
    ),
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}]]
)

# Caso 1
fig1.add_trace(go.Scatter(x=t_plot, y=vm1[:N_PLOT], name="v_m1(t)"), row=1, col=1)
fig1.add_trace(go.Scatter(x=f, y=spec_vm1, name="V_m1(f)"), row=1, col=2)
fig1.update_xaxes(title_text="Tempo (s)", range=[0, T_PLOT_MAX], row=1, col=1)
fig1.update_yaxes(title_text="Amplitude", row=1, col=1)
fig1.update_xaxes(title_text="Frequência (Hz)", range=[0, 5000], row=1, col=2)
fig1.update_yaxes(title_text="Magnitude", row=1, col=2)

# Caso 2
fig1.add_trace(go.Scatter(x=t_plot, y=vm2[:N_PLOT], name="v_m2(t)"), row=2, col=1)
fig1.add_trace(go.Scatter(x=f, y=spec_vm2, name="V_m2(f)"), row=2, col=2)
fig1.update_xaxes(title_text="Tempo (s)", range=[0, T_PLOT_MAX], row=2, col=1)
fig1.update_yaxes(title_text="Amplitude", row=2, col=1)
fig1.update_xaxes(title_text="Frequência (Hz)", range=[0, 5000], row=2, col=2)
fig1.update_yaxes(title_text="Magnitude", row=2, col=2)

# Caso 3
fig1.add_trace(go.Scatter(x=t_plot, y=vm3[:N_PLOT], name="v_m3(t)"), row=3, col=1)
fig1.add_trace(go.Scatter(x=f, y=spec_vm3, name="V_m3(f)"), row=3, col=2)
fig1.update_xaxes(title_text="Tempo (s)", range=[0, T_PLOT_MAX], row=3, col=1)
fig1.update_yaxes(title_text="Amplitude", row=3, col=1)
fig1.update_xaxes(title_text="Frequência (Hz)", range=[0, 5000], row=3, col=2)
fig1.update_yaxes(title_text="Magnitude", row=3, col=2)

fig1.update_layout(
    height=900, title_text="Figura 1: Análise dos Sinais de Mensagem (Banda Base)",
    showlegend=False
)

# Figura 2: Sinal da Portadora
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("v_c(t) - Tempo", "V_c(f) - Frequência"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}]]
)

# Plotar um tempo bem curto para ver as oscilações da portadora
T_PLOT_CARRIER = 0.0005 # 0.5 ms = 5 ciclos da portadora
N_PLOT_CARRIER = int(T_PLOT_CARRIER * F_S)

fig2.add_trace(go.Scatter(x=t[:N_PLOT_CARRIER], y=vc[:N_PLOT_CARRIER], name="v_c(t)"), row=1, col=1)
fig2.add_trace(go.Scatter(x=f, y=spec_vc, name="V_c(f)"), row=1, col=2)
fig2.update_xaxes(title_text="Tempo (s)", range=[0, T_PLOT_CARRIER], row=1, col=1)
fig2.update_yaxes(title_text="Amplitude", row=1, col=1)
fig2.update_xaxes(title_text="Frequência (Hz)", range=[F_C - 1000, F_C + 1000], row=1, col=2) # Zoom na portadora
fig2.update_yaxes(title_text="Magnitude", row=1, col=2)
fig2.update_layout(
    height=400, title_text="Figura 2: Análise da Portadora",
    showlegend=False
)

# Figura 3: Sinais Modulados (Tasks 1.3 e 2.3)
fig3 = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "v_mod1(t) - Tempo", "V_mod1(f) - Frequência",
        "v_mod2(t) - Tempo", "V_mod2(f) - Frequência",
        "v_mod3(t) - Tempo", "V_mod3(f) - Frequência"
    ),
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}]]
)

# Estilo da linha da envoltória
env_style = dict(dash='dash', color='gray', width=1)

# Caso 1
fig3.add_trace(go.Scatter(x=t_plot, y=vmod1[:N_PLOT], name="v_mod1(t)", line=dict(color='blue')), row=1, col=1)
fig3.add_trace(go.Scatter(x=t_plot, y=env1_pos[:N_PLOT], name="Envoltória +", line=env_style), row=1, col=1)
fig3.add_trace(go.Scatter(x=t_plot, y=env1_neg[:N_PLOT], name="Envoltória -", line=env_style), row=1, col=1)
fig3.add_trace(go.Scatter(x=f, y=spec_vmod1, name="V_mod1(f)", line=dict(color='red')), row=1, col=2)
fig3.update_xaxes(title_text="Tempo (s)", range=[0, T_PLOT_MAX], row=1, col=1)
fig3.update_yaxes(title_text="Amplitude", row=1, col=1)
fig3.update_xaxes(title_text="Frequência (Hz)", range=[F_C - 5000, F_C + 5000], row=1, col=2)
fig3.update_yaxes(title_text="Magnitude", row=1, col=2)

# Caso 2
fig3.add_trace(go.Scatter(x=t_plot, y=vmod2[:N_PLOT], name="v_mod2(t)", line=dict(color='blue')), row=2, col=1)
fig3.add_trace(go.Scatter(x=t_plot, y=env2_pos[:N_PLOT], name="Envoltória +", line=env_style), row=2, col=1)
fig3.add_trace(go.Scatter(x=t_plot, y=env2_neg[:N_PLOT], name="Envoltória -", line=env_style), row=2, col=1)
fig3.add_trace(go.Scatter(x=f, y=spec_vmod2, name="V_mod2(f)", line=dict(color='red')), row=2, col=2)
fig3.update_xaxes(title_text="Tempo (s)", range=[0, T_PLOT_MAX], row=2, col=1)
fig3.update_yaxes(title_text="Amplitude", row=2, col=1)
fig3.update_xaxes(title_text="Frequência (Hz)", range=[F_C - 5000, F_C + 5000], row=2, col=2)
fig3.update_yaxes(title_text="Magnitude", row=2, col=2)

# Caso 3
fig3.add_trace(go.Scatter(x=t_plot, y=vmod3[:N_PLOT], name="v_mod3(t)", line=dict(color='blue')), row=3, col=1)
fig3.add_trace(go.Scatter(x=t_plot, y=env3_pos[:N_PLOT], name="Envoltória +", line=env_style), row=3, col=1)
fig3.add_trace(go.Scatter(x=t_plot, y=env3_neg[:N_PLOT], name="Envoltória -", line=env_style), row=3, col=1)
fig3.add_trace(go.Scatter(x=f, y=spec_vmod3, name="V_mod3(f)", line=dict(color='red')), row=3, col=2)
fig3.update_xaxes(title_text="Tempo (s)", range=[0, T_PLOT_MAX], row=3, col=1)
fig3.update_yaxes(title_text="Amplitude", row=3, col=1)
fig3.update_xaxes(title_text="Frequência (Hz)", range=[F_C - 5000, F_C + 5000], row=3, col=2)
fig3.update_yaxes(title_text="Magnitude", row=3, col=2)

fig3.update_layout(
    height=900, title_text="Figura 3: Análise dos Sinais Modulados (AM-LC)",
    showlegend=False
)


print("Gerando gráficos... Verifique as abas do seu navegador.")
fig1.show()
fig2.show()
fig3.show()