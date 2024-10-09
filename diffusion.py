import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def diffusion_1d(C0, D, L, t, nx, nt):
    dx = L / (nx - 1)
    dt = t / (nt - 1)

    x = np.linspace(0, L, nx)
    C = np.zeros((nt, nx))
    C[0, : nx // 2] = C0  # 초기 조건: 왼쪽 절반에만 물질 존재

    for j in range(1, nt):
        for i in range(1, nx - 1):
            C[j, i] = C[j - 1, i] + D * dt / dx**2 * (
                C[j - 1, i + 1] - 2 * C[j - 1, i] + C[j - 1, i - 1]
            )

        C[j, 0] = C[j, 1]  # 경계 조건
        C[j, -1] = C[j, -2]  # 경계 조건

    return x, C


st.title("1차원 확산 시뮬레이션")

C0 = st.slider("초기 농도 (mol/m³)", 0.0, 10.0, 1.0)
D = st.slider("확산 계수 (m²/s)", 1e-5, 1e-4, 5e-5, format="%.1e")
L = st.slider("시스템 길이 (m)", 0.1, 1.0, 0.5)
t = st.slider("총 시뮬레이션 시간 (s)", 1, 100, 50)
nx = 100
nt = 200

if st.button("시뮬레이션 실행"):
    x, C = diffusion_1d(C0, D, L, t, nx, nt)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(0, C0)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Concentration (mol/m³)")

    (line,) = ax.plot([], [], lw=2)

    progress_bar = st.progress(0)
    plot_placeholder = st.empty()

    for i in range(nt):
        progress_bar.progress(i / (nt - 1))

        line.set_data(x, C[i, :])
        ax.set_title(f"1D Diffusion Profile (Time: {i*t/nt:.2f} s)")

        plot_placeholder.pyplot(fig)
        plt.close(fig)

    st.write(f"확산 계수: {D} m²/s")
    st.write(f"시스템 길이: {L} m")
    st.write(f"총 시뮬레이션 시간: {t} s")

    st.write("### 미분방정식")
    st.latex(r"\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2}")
    st.write("여기서:")
    st.write("- C는 농도 (mol/m³)")
    st.write("- t는 시간 (s)")
    st.write("- D는 확산 계수 (m²/s)")
    st.write("- x는 위치 (m)")

    st.write("### 시뮬레이션의 의미:")
    st.write("1. 초기 조건: 처음에는 물질이 시스템의 왼쪽 절반에만 존재합니다.")
    st.write(
        "2. 시간 경과: 시간이 지남에 따라 물질은 농도가 높은 영역에서 낮은 영역으로 이동합니다."
    )
    st.write(
        "3. 농도 구배: 그래프의 기울기는 농도 구배를 나타내며, 이는 확산의 원동력입니다."
    )
    st.write(
        "4. 평형 상태: 충분한 시간이 지나면 시스템은 균일한 농도 분포에 도달하려 합니다."
    )
    st.write("5. 확산 계수의 영향: 확산 계수가 클수록 물질이 더 빠르게 확산됩니다.")

st.write(
    "이 시뮬레이션은 화학공정에서 중요한 물질 전달 현상인 확산을 모델링합니다. 실제 공정에서는 이러한 원리를 이용하여 혼합, 분리, 반응 속도 제어 등 다양한 응용이 가능합니다."
)
