import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
import math

st.set_page_config(page_title="Filtro Mosaico con Círculos/Estrellas", layout="wide")


def calcular_color_promedio(imagen, x0, y0, x1, y1):
    """
    Calcula el color promedio (R, G, B) de la región definida por
    las esquinas (x0, y0) y (x1, y1) en la imagen dada.
    """
    # Para acceso rápido a los píxeles
    px = imagen.load()

    # Acumuladores
    r_total, g_total, b_total = 0, 0, 0
    contador = 0

    for y in range(y0, y1):
        for x in range(x0, x1):
            r, g, b = px[x, y]
            r_total += r
            g_total += g
            b_total += b
            contador += 1

    if contador == 0:
        return (0, 0, 0)

    return (
        r_total // contador,
        g_total // contador,
        b_total // contador
    )


def generar_estrella(cx, cy, r_ext, r_int, num_puntas=5):
    """
    Genera la lista de coordenadas (x, y) para dibujar una estrella.
    - cx, cy: centro de la estrella
    - r_ext: radio exterior (punta de la estrella)
    - r_int: radio interior (valle entre puntas)
    - num_puntas: cuántas puntas tendrá la estrella
    Retorna una lista de tuplas (x, y).
    """
    puntos = []
    angulo_inicial = -math.pi / 2  # para que la estrella apunte hacia arriba
    angulo_por_punta = math.pi / num_puntas

    for i in range(num_puntas * 2):
        # Alterna entre radio exterior e interior
        if i % 2 == 0:
            r = r_ext
        else:
            r = r_int

        # Ángulo correspondiente a este vértice
        ang = angulo_inicial + i * angulo_por_punta

        x = cx + int(r * math.cos(ang))
        y = cy + int(r * math.sin(ang))
        puntos.append((x, y))

    return puntos


def aplicar_mosaico_formas(imagen, cell_size=20, forma="Circulos"):
    """
    Aplica el filtro de mosaico dividiendo la imagen en celdas de tamaño cell_size
    y sustituyendo cada celda por una figura (círculo o estrella) de color promedio.
    """
    width, height = imagen.size

    # Convertimos a RGB por seguridad (a veces las imágenes pueden venir en RGBA)
    imagen_rgb = imagen.convert("RGB")

    # Crearemos una nueva imagen "en blanco" (fondo blanco)
    salida = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(salida)

    # Recorremos la imagen en pasos de cell_size
    for y in range(0, height, cell_size):
        for x in range(0, width, cell_size):
            # Coordenadas de la región de la celda
            x1 = min(x + cell_size, width)
            y1 = min(y + cell_size, height)

            # Calculamos el color promedio de esta región
            color_prom = calcular_color_promedio(imagen_rgb, x, y, x1, y1)

            # Centro de la celda (para ubicar la forma)
            cx = x + (cell_size // 2)
            cy = y + (cell_size // 2)

            # Dibujamos la figura correspondiente
            if forma == "Circulos":
                # Radio del círculo (reducimos un poco para que se note separación)
                radio = cell_size // 2
                bbox = (cx - radio, cy - radio, cx + radio, cy + radio)
                draw.ellipse(bbox, fill=color_prom, outline=None)

            elif forma == "Estrellas":
                # Generar estrella con 5 puntas
                # (r_ext ~ radio externo, r_int ~ radio interno)
                r_ext = cell_size // 2  # tamaño "grande"
                r_int = cell_size // 4  # tamaño "pequeño" para valles
                puntos_estrella = generar_estrella(cx, cy, r_ext, r_int, num_puntas=5)
                draw.polygon(puntos_estrella, fill=color_prom)

    return salida


def main():
    st.sidebar.title("Configuraciones del Mosaico")

    # Parámetros
    cell_size = st.sidebar.slider("Tamaño de cada mosaico (px)", 5, 80, 20, step=5)
    forma = st.sidebar.selectbox("Forma del mosaico", ["Circulos", "Estrellas"])

    uploaded_file = st.sidebar.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    imagen_resultante = None
    buf_value = None

    if uploaded_file is not None:
        # Cargamos la imagen original
        imagen_original = Image.open(uploaded_file)

        with st.spinner(f"Aplicando filtro de mosaico con {forma}..."):
            imagen_resultante = aplicar_mosaico_formas(imagen_original, cell_size, forma)

        # Preparar la imagen para descarga
        buf = BytesIO()
        imagen_resultante.save(buf, format="PNG")
        buf_value = buf.getvalue()

    # Título y botón de descarga
    title_col, button_col = st.columns([4, 1])

    with title_col:
        st.title("Filtro de Mosaico")
    with button_col:
        if imagen_resultante is not None and buf_value is not None:
            st.download_button(
                label="⬇️ Descargar imagen",
                data=buf_value,
                file_name="imagen_mosaico.png",
                mime="image/png",
                key="download_button_top"
            )

    # Mostrar las imágenes si se subió un archivo
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(imagen_original, caption="Imagen Original", use_container_width=True)
        with col2:
            st.image(imagen_resultante, caption=f"Imagen con Mosaico de {forma}", use_container_width=True)
    else:
        st.info("Sube una imagen para aplicar el filtro de mosaico con círculos o estrellas.")


if __name__ == "__main__":
    main()
