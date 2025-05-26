import io
import threading
import time
from functools import lru_cache
from typing import Tuple, List

import numpy as np
import streamlit as st
from PIL import Image


class RecursiveImageProcessor:
    """
    Clase que implementa el procesamiento de imágenes recursivas con mayor eficiencia
    y calidad visual, especialmente para casos donde se requiere zoom de alta resolución.
    """

    def __init__(self, image: Image.Image):
        """
        Inicializa el procesador con una imagen de entrada.

        Args:
            image: Imagen PIL en formato RGB
        """
        self.original_image = image.convert('RGB')
        # Convertimos la imagen a NumPy array para procesamiento más rápido
        self.image_array = np.array(self.original_image)
        self.width, self.height = self.original_image.size
        # Precalculamos el color promedio de toda la imagen
        self.global_avg_color = self._compute_image_average_color()
        # Cache para los colores promedio por bloque
        self.block_color_cache = {}

    def _compute_image_average_color(self) -> Tuple[int, int, int]:
        """
        Calcula el color promedio de toda la imagen usando NumPy para mejor rendimiento.

        Returns:
            Tupla con los valores RGB promedio
        """
        # Cálculo vectorizado usando NumPy
        avg_color = np.mean(self.image_array, axis=(0, 1), dtype=np.int32)
        return tuple(avg_color)

    @lru_cache(maxsize=1024)
    def _compute_block_average_color(self, x0: int, y0: int, block_size: int) -> Tuple[int, int, int]:
        """
        Calcula el color promedio de un bloque específico con caché para mejorar el rendimiento.

        Args:
            x0: Coordenada X inicial del bloque
            y0: Coordenada Y inicial del bloque
            block_size: Tamaño del bloque en píxeles

        Returns:
            Tupla con los valores RGB promedio del bloque
        """
        # Límites para evitar acceder fuera de los límites de la imagen
        x_end = min(x0 + block_size, self.width)
        y_end = min(y0 + block_size, self.height)

        # Si el bloque está completamente fuera de la imagen
        if x0 >= self.width or y0 >= self.height or x_end <= x0 or y_end <= y0:
            return (0, 0, 0)

        # Extraer el bloque como un subarray y calcular el promedio
        block = self.image_array[y0:y_end, x0:x_end]
        avg_color = np.mean(block, axis=(0, 1), dtype=np.int32)
        return tuple(avg_color)

    def _recolor_image(self,
                       image: Image.Image,
                       target_avg: Tuple[int, int, int],
                       block_avg: Tuple[int, int, int]) -> Image.Image:
        """
        Re-colorea una imagen para que su color promedio se aproxime al bloque original.
        Implementación mejorada con NumPy para mayor velocidad y calidad.

        Args:
            image: Imagen PIL a recolorear
            target_avg: Color promedio de la imagen a recolorear
            block_avg: Color promedio del bloque en la imagen original

        Returns:
            Imagen recoloreada
        """
        # Desempaquetar colores promedio
        tr, tg, tb = target_avg
        br, bg, bb = block_avg

        # Evitar divisiones por cero
        tr = max(1, tr)
        tg = max(1, tg)
        tb = max(1, tb)

        # Convertir a NumPy array para procesamiento vectorizado
        img_array = np.array(image)

        # Factores de escalado para cada canal
        factor_r = br / tr
        factor_g = bg / tg
        factor_b = bb / tb

        # Aplicar transformación de color vectorizada
        # Creamos una copia para no modificar el original
        recolored_array = img_array.copy()

        # Multiplica cada canal por su factor correspondiente
        recolored_array[:, :, 0] = np.clip(recolored_array[:, :, 0] * factor_r, 0, 255).astype(np.uint8)
        recolored_array[:, :, 1] = np.clip(recolored_array[:, :, 1] * factor_g, 0, 255).astype(np.uint8)
        recolored_array[:, :, 2] = np.clip(recolored_array[:, :, 2] * factor_b, 0, 255).astype(np.uint8)

        return Image.fromarray(recolored_array)

    def process_image(self,
                      block_size: int = 10,
                      quality_level: str = "normal",
                      show_progress: bool = True) -> Image.Image:
        """
        Crea una imagen recursiva con diferentes niveles de calidad.

        Args:
            block_size: Tamaño del bloque en píxeles
            quality_level: Nivel de calidad ('low', 'normal', 'high', 'ultra')
            show_progress: Si se debe mostrar una barra de progreso

        Returns:
            Imagen procesada con el efecto recursivo
        """
        # Configurar la calidad según el nivel elegido
        if quality_level == "low":
            scale_factor = 0.75
            resampling = Image.Resampling.BILINEAR
        elif quality_level == "normal":
            scale_factor = 1.0
            resampling = Image.Resampling.LANCZOS
        elif quality_level == "high":
            scale_factor = 1.5
            resampling = Image.Resampling.LANCZOS
        elif quality_level == "ultra":
            scale_factor = 2.0
            resampling = Image.Resampling.LANCZOS

        # Calcular el tamaño del bloque ajustado
        adjusted_block_size = max(1, int(block_size * scale_factor))

        # Crear imagen de salida
        output_width = int(self.width * scale_factor)
        output_height = int(self.height * scale_factor)
        new_image = Image.new("RGB", (output_width, output_height))

        # Para procesamiento paralelo, dividimos la imagen en secciones
        sections = self._divide_into_sections(output_width, output_height, adjusted_block_size)

        # Configurar la barra de progreso
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()

        # Escalar la imagen original al tamaño del bloque para las miniaturas
        scaled_image = self.original_image.resize(
            (adjusted_block_size, adjusted_block_size),
            resampling
        )

        # Color promedio de la imagen escalada
        scaled_avg_color = self._compute_image_average_color()

        # Procesar por secciones (potencialmente en paralelo)
        results = []
        threads = []

        # Crear hilos para procesamiento paralelo
        for i, section in enumerate(sections):
            thread = threading.Thread(
                target=self._process_section,
                args=(
                    new_image, section, adjusted_block_size,
                    scaled_image, scaled_avg_color, results, i
                )
            )
            threads.append(thread)
            thread.start()

        # Actualizar la barra de progreso mientras se procesan los hilos
        if show_progress:
            while any(thread.is_alive() for thread in threads):
                completed = len([r for r in results if r is not None])
                progress = completed / len(sections)
                progress_bar.progress(progress)
                status_text.text(f"Procesando imagen: {int(progress * 100)}%")
                time.sleep(0.1)

            # Completar la barra de progreso
            progress_bar.progress(1.0)
            status_text.text("¡Procesamiento completado!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

        # Esperar a que todos los hilos terminen
        for thread in threads:
            thread.join()

        return new_image

    def _divide_into_sections(self,
                              width: int,
                              height: int,
                              block_size: int) -> List[Tuple[int, int, int, int]]:
        """
        Divide la imagen en secciones para procesamiento paralelo.

        Args:
            width: Ancho de la imagen
            height: Alto de la imagen
            block_size: Tamaño del bloque

        Returns:
            Lista de secciones (x_start, y_start, x_end, y_end)
        """
        # Determinar el número óptimo de secciones basado en los núcleos disponibles
        # En un caso real, se podría usar multiprocessing.cpu_count()
        num_sections = 4  # Valor razonable para la mayoría de sistemas

        sections = []
        section_height = height // num_sections

        for i in range(num_sections):
            y_start = i * section_height
            y_end = height if i == num_sections - 1 else (i + 1) * section_height
            sections.append((0, y_start, width, y_end))

        return sections

    def _process_section(self,
                         new_image: Image.Image,
                         section: Tuple[int, int, int, int],
                         block_size: int,
                         scaled_image: Image.Image,
                         scaled_avg_color: Tuple[int, int, int],
                         results: list,
                         index: int) -> None:
        """
        Procesa una sección de la imagen (para paralelización).

        Args:
            new_image: Imagen de destino
            section: Coordenadas de la sección (x_start, y_start, x_end, y_end)
            block_size: Tamaño del bloque
            scaled_image: Imagen miniatura escalada
            scaled_avg_color: Color promedio de la imagen escalada
            results: Lista para almacenar resultados
            index: Índice de la sección para registro
        """
        x_start, y_start, x_end, y_end = section

        # Crear un lienzo temporal para esta sección
        section_width = x_end - x_start
        section_height = y_end - y_start
        section_image = Image.new("RGB", (section_width, section_height))

        # Ajustar coordenadas al tamaño original
        scale_factor = self.width / new_image.width

        # Procesar cada bloque en esta sección
        for y in range(y_start, y_end, block_size):
            for x in range(x_start, x_end, block_size):
                # Mapear coordenadas a la imagen original
                orig_x = int(x * scale_factor)
                orig_y = int(y * scale_factor)

                # Calcular el color promedio del bloque correspondiente en la imagen original
                block_avg_color = self._compute_block_average_color(
                    orig_x, orig_y, int(block_size * scale_factor)
                )

                # Recolorear la imagen miniatura
                recolored_tile = self._recolor_image(
                    scaled_image, scaled_avg_color, block_avg_color
                )

                # Pegar en la posición correcta relativa a la sección
                section_image.paste(
                    recolored_tile, (x - x_start, y - y_start)
                )

        # Pegar la sección procesada en la imagen final
        new_image.paste(section_image, (x_start, y_start))

        # Marcar esta sección como completada
        results[index] = True


def create_ui():
    """
    Crea la interfaz de usuario de Streamlit con mejoras visuales y funcionales.
    """
    # Configuración de la página
    st.set_page_config(
        page_title="Filtro de Imagen Recursiva",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Estilo personalizado
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1E88E5;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Barra lateral con controles
    with st.sidebar:
        st.title("Imagen Recursiva")

        st.markdown("""
        Este filtro crea un mosaico 'fractal' con la misma imagen, 
        reemplazando cada bloque con una versión miniatura de la imagen completa.
        """)

        # Carga de archivo con soporte para más formatos
        uploaded_file = st.file_uploader(
            "Sube una imagen",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            help="Soporta varios formatos de imagen"
        )

        # Agrupamos controles relacionados
        with st.expander("Configuración del Filtro", expanded=True):
            # Parámetro: tamaño de bloque con más contexto
            block_size = st.slider(
                "Tamaño del bloque (px)",
                min_value=4,
                max_value=64,
                value=16,
                step=4,
                help="Bloques más pequeños crean más detalles, bloques más grandes muestran mejor el efecto recursivo"
            )

            # Calidad de procesamiento
            quality_options = {
                "low": "Baja (más rápido)",
                "normal": "Normal",
                "high": "Alta (mejor zoom)",
                "ultra": "Ultra (máxima calidad)"
            }

            quality_level = st.select_slider(
                "Calidad de procesamiento",
                options=list(quality_options.keys()),
                value="normal",
                format_func=lambda x: quality_options[x],
                help="Mayor calidad permite mejor zoom pero requiere más tiempo de procesamiento"
            )

        # Información adicional
        with st.expander("Acerca del Algoritmo", expanded=False):
            st.markdown("""
            ### ¿Cómo funciona?

            1. Divide la imagen en pequeños bloques
            2. Para cada bloque:
               - Calcula su color promedio
               - Reemplaza con una miniatura de la imagen completa
               - Ajusta el color de la miniatura para que coincida con el bloque original

            El resultado es un efecto "fractaloide" donde la imagen se contiene a sí misma.
            """)

    # Área principal
    st.title("Filtro de Imagen Recursiva")
    st.markdown("**Transforma tus imágenes con un efecto recursivo de alta resolución**")

    # Manejo de la imagen cargada
    if uploaded_file is not None:
        try:
            # Cargar la imagen y mostrar información
            original_image = Image.open(uploaded_file)

            # Información sobre la imagen
            width, height = original_image.size
            st.info(f"Imagen cargada: {width}x{height} píxeles - {original_image.format}")

            # Configurar columnas para la visualización
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Imagen Original")
                st.image(
                    original_image,
                    use_container_width=True,
                    caption="Imagen de entrada"
                )

            with col2:
                st.subheader("Vista Previa")

                # Inicializar el procesador
                processor = RecursiveImageProcessor(original_image)

                # Botón de procesamiento con estado
                process_button = st.button(
                    "📸 Procesar Imagen",
                    help="Haz clic para aplicar el filtro recursivo con los parámetros actuales"
                )

                if process_button or 'result_image' in st.session_state:
                    # Mostrar aviso de procesamiento
                    with st.spinner("Procesando imagen... Por favor espera"):
                        if process_button:
                            # Crear la imagen procesada
                            result_image = processor.process_image(
                                block_size=block_size,
                                quality_level=quality_level,
                                show_progress=True
                            )
                            # Guardar en el estado de la sesión
                            st.session_state['result_image'] = result_image
                        else:
                            # Recuperar del estado de la sesión
                            result_image = st.session_state['result_image']

                    # Mostrar la imagen resultante
                    st.image(
                        result_image,
                        use_container_width=True,
                        caption=f"Filtro recursivo (bloques de {block_size}px, calidad {quality_level})"
                    )

                    # Botón de descarga
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")

                    st.download_button(
                        label="⬇️ Descargar Imagen Procesada",
                        data=buf.getvalue(),
                        file_name=f"recursivo_{block_size}px_{quality_level}.png",
                        mime="image/png",
                        help="Descarga la imagen procesada en formato PNG"
                    )
                else:
                    # Mensaje informativo
                    st.info("Haz clic en 'Procesar Imagen' para aplicar el filtro recursivo.")

        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")
            st.exception(e)
    else:
        # Mostrar información de inicio
        st.markdown("""
        ## 👋 ¡Bienvenido al Filtro de Imagen Recursiva!

        Para comenzar:
        1. Sube una imagen usando el selector en la barra lateral
        2. Ajusta los parámetros del filtro según tus preferencias
        3. Haz clic en "Procesar Imagen" para ver el resultado

        El filtro creará un efecto donde cada parte de la imagen es reemplazada por una versión miniatura de la imagen completa.
        """)


def main():
    """Función principal que inicializa la aplicación."""
    try:
        create_ui()
    except Exception as e:
        st.error("Ocurrió un error inesperado en la aplicación.")
        st.exception(e)


# Punto de entrada
if __name__ == "__main__":
    main()