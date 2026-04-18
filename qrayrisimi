import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Sayfa yapılandırması
st.set_page_config(page_title="QR Ayrışımı ile En Küçük Kareler", layout="wide")

st.title("📊 En Küçük Kareler Yöntemi ve QR Ayrışımı")
st.markdown("""
Bu interaktif uygulamada, rastgele belirlenmiş 5 veri noktasına en uygun doğruyu (**Linear Regression**) 
**QR Ayrışımı** yöntemini kullanarak hesaplıyoruz.
""")

# --- SIDEBAR (YAN MENÜ) ---
st.sidebar.header("Veri Noktalarını Ayarla")
st.sidebar.write("Y değerlerini değiştirerek grafiği anlık güncelleyin:")

# Sabit X değerleri (0, 1, 2, 3, 4)
x_vals = np.array([0, 1, 2, 3, 4])
y_vals = []

# Kullanıcıdan 5 adet Y değeri alımı
for i in range(5):
    val = st.sidebar.slider(f"Nokta {i+1} (x={i})", min_value=0.0, max_value=10.0, value=float(i*1.5 + 2), step=0.1)
    y_vals.append(val)

y_vals = np.array(y_vals)

# --- MATEMATİKSEL HESAPLAMA (QR AYRIŞIMI) ---
# Model: y = a + bx
# Tasarım matrisi A'yı oluşturuyoruz: [1, x]
A = np.vstack([np.ones(len(x_vals)), x_vals]).T

# 1. A = QR Ayrışımı (Numpy ile)
Q, R = np.linalg.qr(A)

# 2. En Küçük Kareler Çözümü: R * beta = Q^T * y
# Üst üçgen matris çözümü (Back-substitution)
beta = np.linalg.solve(R, Q.T @ y_vals)
sabit_terim, egim = beta

# Tahmin edilen Y değerleri
y_pred = A @ beta

# --- GÖRSELLEŞTİRME (PLOTLY) ---
fig = go.Figure()

# Gerçek Veri Noktaları
fig.add_trace(go.Scatter(
    x=x_vals, y=y_vals,
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Gerçek Veriler'
))

# Uydurulan Doğru (Regression Line)
x_range = np.linspace(-0.5, 4.5, 100)
y_range = sabit_terim + egim * x_range
fig.add_trace(go.Scatter(
    x=x_range, y=y_range,
    mode='lines',
    line=dict(color='blue', width=3),
    name='QR ile Uydurulan Doğru'
))

# Hata Çizgileri (Residuals)
for i in range(len(x_vals)):
    fig.add_trace(go.Scatter(
        x=[x_vals[i], x_vals[i]],
        y=[y_vals[i], y_pred[i]],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False
    ))

fig.update_layout(
    title="Doğrusal Regresyon ve Hata Sapmaları",
    xaxis_title="X Ekseni",
    yaxis_title="Y Ekseni",
    template="plotly_white",
    height=600
)

# --- EKRAN ÇIKTILARI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Hesaplama Sonuçları")
    st.info(f"**Elde Edilen Denklem:**\n\n $y = {sabit_terim:.3f} + {egim:.3f}x$")
    
    st.write("---")
    st.write("**Matematiksel Özet:**")
    st.latex(r"A = QR")
    st.latex(r"R\hat{x} = Q^T y")
    
    with st.expander("A Matrisini Gör"):
        st.write(A)
    with st.expander("R Matrisini (Üst Üçgen) Gör"):
        st.write(R)

st.success("Veriler anlık olarak QR ayrışımı kullanılarak optimize edilmiştir.")
