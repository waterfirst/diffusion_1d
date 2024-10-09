import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def diffusion_2d(C0, D, Lx, Ly, t, nx, ny, nt):
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = t / (nt - 1)

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    C = np.zeros((nt, ny, nx))
    C[0, ny // 4 : 3 * ny // 4, nx // 4 : 3 * nx // 4] = (
        C0  # 초기 조건: 중앙에 정사각형 모양으로 물질 존재
    )

    for k in range(1, nt):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                C[k, i, j] = C[k - 1, i, j] + D * dt * (
                    (C[k - 1, i + 1, j] - 2 * C[k - 1, i, j] + C[k - 1, i - 1, j])
                    / dx**2
                    + (C[k - 1, i, j + 1] - 2 * C[k - 1, i, j] + C[k - 1, i, j - 1])
                    / dy**2
                )

        # 경계 조건 (Neumann 경계 조건)
        C[k, 0, :] = C[k, 1, :]
        C[k, -1, :] = C[k, -2, :]
        C[k, :, 0] = C[k, :, 1]
        C[k, :, -1] = C[k, :, -2]

    return x, y, C


st.title("2차원 확산 시뮬레이션 (2D 및 3D 시각화)")

C0 = st.slider("초기 농도 (mol/m³)", 0.0, 10.0, 1.0)
D = st.slider("확산 계수 (m²/s)", 1e-5, 1e-4, 5e-5, format="%.1e")
L = st.slider("시스템 크기 (m)", 0.1, 1.0, 0.5)
t = st.slider("총 시뮬레이션 시간 (s)", 1, 100, 50)
interval = st.slider("프레임 간격 (ms)", 10, 1000, 100)
nx, ny, nt = 50, 50, 100

if st.button("시뮬레이션 실행"):
    x, y, C = diffusion_2d(C0, D, L, L, t, nx, ny, nt)

    # 2D 그래프 설정
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.imshow(
        C[0], extent=[0, L, 0, L], origin="lower", vmin=0, vmax=C0, cmap="viridis"
    )
    plt.colorbar(im1, ax=ax1, label="Concentration (mol/m³)")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    title1 = ax1.set_title("2D Diffusion Profile (Time: 0.00 s)")

    # 3D 그래프 설정
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(x, y)
    surf = ax2.plot_surface(X, Y, C[0], cmap="viridis", vmin=0, vmax=C0)
    fig2.colorbar(surf, ax=ax2, label="Concentration (mol/m³)")
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.set_zlabel("Concentration (mol/m³)")
    title2 = ax2.set_title("3D Diffusion Profile (Time: 0.00 s)")

    plot_placeholder1 = st.empty()
    plot_placeholder2 = st.empty()

    for i in range(nt):
        # 2D 그래프 업데이트
        im1.set_array(C[i])
        title1.set_text(f"2D Diffusion Profile (Time: {i*t/nt:.2f} s)")
        plot_placeholder1.pyplot(fig1)

        # 3D 그래프 업데이트
        ax2.clear()
        surf = ax2.plot_surface(X, Y, C[i], cmap="viridis", vmin=0, vmax=C0)
        ax2.set_xlabel("X Position (m)")
        ax2.set_ylabel("Y Position (m)")
        ax2.set_zlabel("Concentration (mol/m³)")
        title2 = ax2.set_title(f"3D Diffusion Profile (Time: {i*t/nt:.2f} s)")
        plot_placeholder2.pyplot(fig2)

        time.sleep(interval / 1000)  # 프레임 간격을 초 단위로 변환

    st.write(f"확산 계수: {D} m²/s")
    st.write(f"시스템 크기: {L}x{L} m")
    st.write(f"총 시뮬레이션 시간: {t} s")

    st.write("### 미분방정식")
    st.latex(
        r"\frac{\partial C}{\partial t} = D \left(\frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2}\right)"
    )
    st.write("여기서:")
    st.write("- C는 농도 (mol/m³)")
    st.write("- t는 시간 (s)")
    st.write("- D는 확산 계수 (m²/s)")
    st.write("- x, y는 위치 (m)")

    st.write("### 시뮬레이션의 의미:")
    st.write(
        "1. 초기 조건: 처음에는 물질이 시스템의 중앙 부분에 정사각형 모양으로 존재합니다."
    )
    st.write(
        "2. 시간 경과: 시간이 지남에 따라 물질은 농도가 높은 영역에서 낮은 영역으로 모든 방향으로 이동합니다."
    )
    st.write(
        "3. 농도 구배: 색상의 변화와 3D 그래프의 높이는 농도 구배를 나타내며, 이는 확산의 원동력입니다."
    )
    st.write(
        "4. 대칭성: 초기 조건과 경계 조건이 대칭적이므로, 확산 패턴도 대칭적으로 나타납니다."
    )
    st.write(
        "5. 평형 상태: 충분한 시간이 지나면 시스템은 균일한 농도 분포에 도달하려 합니다."
    )
    st.write("6. 확산 계수의 영향: 확산 계수가 클수록 물질이 더 빠르게 확산됩니다.")

st.write(
    "이 2차원 확산 시뮬레이션은 평면 상에서의 물질 이동을 2D와 3D로 모델링합니다. 실제 공정에서는 이러한 원리를 이용하여 막 분리, 표면 처리, 촉매 반응 등 다양한 응용이 가능합니다."
)
