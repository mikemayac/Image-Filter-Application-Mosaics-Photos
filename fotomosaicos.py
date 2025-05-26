import io
import os
import threading
import time
import math
from functools import lru_cache
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, UnidentifiedImageError


# --- Funciones Helper Estáticas ---
def _compute_average_color_static(image_array: np.ndarray) -> Tuple[int, int, int]:
    if image_array.size == 0:
        return (0, 0, 0)
    if image_array.ndim == 2:
        avg_color_val = int(np.mean(image_array, dtype=np.int32))
        return (avg_color_val, avg_color_val, avg_color_val)
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        avg_color_val = int(np.mean(image_array[:, :, 0], dtype=np.int32))
        return (avg_color_val, avg_color_val, avg_color_val)
    elif image_array.ndim == 3 and image_array.shape[2] == 4:  # RGBA
        avg_rgb = np.mean(image_array[:, :, :3], axis=(0, 1), dtype=np.int32)
        return tuple(c if c <= 255 else 255 for c in avg_rgb)  # Asegurar que no exceda 255
    elif image_array.ndim == 3 and image_array.shape[2] == 3:  # RGB
        avg_color = np.mean(image_array, axis=(0, 1), dtype=np.int32)
        return tuple(c if c <= 255 else 255 for c in avg_color)  # Asegurar que no exceda 255
    else:
        return (0, 0, 0)


@st.cache_data(show_spinner="Procesando biblioteca de imágenes... Por favor espera.")
def get_processed_library_thumbnails(_lib_path_key: str, library_path: str, block_size: int) -> List[Dict[str, Any]]:
    processed_thumbnails = []
    if not library_path or not os.path.isdir(library_path):
        print(f"Error cacheado: Ruta de biblioteca no válida o no es un directorio: {library_path}")
        return []

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    try:
        image_files = [f for f in os.listdir(library_path)
                       if os.path.isfile(os.path.join(library_path, f)) and f.lower().endswith(valid_extensions)]
    except FileNotFoundError:
        print(f"Error cacheado: Directorio de biblioteca no encontrado: {library_path}")
        return []

    if not image_files:
        print(f"Error cacheado: No se encontraron imágenes válidas en la biblioteca: {library_path}")
        return []

    for i, filename in enumerate(image_files):
        try:
            img_path = os.path.join(library_path, filename)
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
                thumbnail = img_rgb.resize((block_size, block_size), Image.Resampling.LANCZOS)
                avg_color = _compute_average_color_static(np.array(thumbnail))
                processed_thumbnails.append({'image_pil': thumbnail, 'avg_color': avg_color, 'path': img_path})
        except FileNotFoundError:
            print(f"Error cacheado: Archivo no encontrado: {filename}")
        except UnidentifiedImageError:
            print(f"Error cacheado: Imagen corrupta/no identificable (saltando): {filename}")
        except Exception as e:
            print(f"Error cacheado: Procesando {filename} de la biblioteca: {e} (saltando)")

    if not processed_thumbnails:
        print(f"Error cacheado: No se pudieron cargar imágenes válidas de la biblioteca.")

    return processed_thumbnails


# --- Clase Principal del Procesador de Fotomosaicos ---
class PhotomosaicProcessor:
    def __init__(self, target_image: Image.Image, block_size: int, metric: str = "euclidean"):
        self.target_image = target_image.convert('RGB')  # El realce de contraste se aplica antes
        self.target_image_array = np.array(self.target_image)
        self.width, self.height = self.target_image.size
        self.block_size = block_size
        self.library_thumbnails: List[Dict[str, Any]] = []
        self.metric = metric

    @lru_cache(maxsize=8192)  # Aumentar caché para bloques de imagen objetivo
    def _get_target_block_average_color(self, x0: int, y0: int) -> Tuple[int, int, int]:
        x_end = min(x0 + self.block_size, self.width)
        y_end = min(y0 + self.block_size, self.height)
        if x0 >= self.width or y0 >= self.height or x_end <= x0 or y_end <= y0:
            return (0, 0, 0)
        block_array = self.target_image_array[y0:y_end, x0:x_end]
        return _compute_average_color_static(block_array)

    def _calculate_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        r1, g1, b1 = float(color1[0]), float(color1[1]), float(color1[2])
        r2, g2, b2 = float(color2[0]), float(color2[1]), float(color2[2])

        if self.metric == "linear":
            num_color1 = (65535.0 * r1) + (256.0 * g1) + b1
            num_color2 = (65535.0 * r2) + (256.0 * g2) + b2
            return abs(num_color1 - num_color2)

        elif self.metric == "riemersma":
            # Usamos la fórmula del PDF, sin sqrt para mantener "distancia al cuadrado"
            r_avg = (r1 + r2) / 2.0
            dr_sq = (r1 - r2) ** 2
            dg_sq = (g1 - g2) ** 2
            db_sq = (b1 - b2) ** 2
            # Evitar división por cero o valores negativos de r_avg si los colores no son 0-255
            # Aunque PIL asegura 0-255 para RGB.
            term_r = (2.0 + max(0, r_avg) / 256.0) * dr_sq
            term_g = 4.0 * dg_sq
            term_b = (2.0 + max(0, (255.0 - r_avg)) / 256.0) * db_sq
            return term_r + term_g + term_b

        # Por defecto o si es "euclidean" (distancia euclidiana al cuadrado)
        else:
            return (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2

    def _find_best_match_tile_item(self, target_block_avg_color: Tuple[int, int, int],
                                   last_selected_path: Optional[str] = None,
                                   allow_adjacent_repeats: bool = True) -> Optional[Dict[str, Any]]:
        if not self.library_thumbnails:
            return None

        min_distance = float('inf')
        best_item = None

        for item in self.library_thumbnails:
            if not allow_adjacent_repeats and last_selected_path and item['path'] == last_selected_path:
                continue  # Saltar si es el mismo que el anterior y no se permiten repeticiones adyacentes

            distance = self._calculate_distance(target_block_avg_color, item['avg_color'])

            if distance < min_distance:
                min_distance = distance
                best_item = item
            if min_distance == 0:  # Coincidencia perfecta
                break

        # Si no se encontró ninguno (ej. todos fueron excluidos), devolver el primero como fallback
        if not best_item and self.library_thumbnails:
            return self.library_thumbnails[0]

        return best_item

    def _divide_into_sections(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        num_physical_cores = os.cpu_count() or 4
        # Limitar secciones para no crear demasiados hilos pequeños
        max_sections_based_on_rows = height // self.block_size if self.block_size > 0 else height
        num_sections = min(num_physical_cores, max_sections_based_on_rows)
        num_sections = max(1, num_sections)

        sections = []
        pixels_per_section_approx = height // num_sections

        current_y = 0
        for i in range(num_sections):
            y_s = current_y
            if i == num_sections - 1:  # Última sección toma todo lo restante
                y_e = height
            else:
                # Alinear y_e al múltiplo más cercano de block_size, o casi
                num_rows_in_section = max(1, round(
                    pixels_per_section_approx / self.block_size if self.block_size > 0 else pixels_per_section_approx))
                y_e = y_s + num_rows_in_section * (self.block_size if self.block_size > 0 else 1)
                y_e = min(y_e, height)  # No exceder

            if y_s < y_e:
                sections.append((0, y_s, width, y_e))
            current_y = y_e
            if current_y >= height: break
        return sections

    def _process_section(self, new_image: Image.Image, section_coords: Tuple[int, int, int, int],
                         results_list: list, section_idx: int, allow_adjacent_repeats: bool) -> None:
        x_start, y_start, x_end, y_end = section_coords

        last_tile_path_in_row: Optional[str] = None

        for y_global in range(y_start, y_end, self.block_size):
            # Resetear last_tile_path_in_row para cada nueva fila en la sección
            # (la no repetición adyacente es solo horizontal aquí)
            last_tile_path_in_row = None
            for x_global in range(x_start, x_end, self.block_size):
                # Coordenadas relativas al canvas de la imagen final (new_image)
                # Asegurar que estamos dentro de los límites efectivos del output_width/height
                if x_global < new_image.width and y_global < new_image.height:
                    target_avg_color = self._get_target_block_average_color(x_global, y_global)

                    best_item = self._find_best_match_tile_item(target_avg_color, last_tile_path_in_row,
                                                                allow_adjacent_repeats)

                    if best_item:
                        best_tile_img = best_item['image_pil']
                        new_image.paste(best_tile_img, (x_global, y_global))
                        if not allow_adjacent_repeats:
                            last_tile_path_in_row = best_item['path']
                    else:  # Fallback si no se encuentra ninguna tesela (biblioteca vacía o todas excluidas)
                        placeholder_tile = Image.new("RGB", (self.block_size, self.block_size), (128, 128, 128))
                        new_image.paste(placeholder_tile, (x_global, y_global))

        results_list[section_idx] = True

    def create_photomosaic(self, allow_adjacent_repeats: bool,
                           show_progress_st_elements: Optional[Tuple[Any, Any]] = None) -> Image.Image:
        if not self.library_thumbnails:
            error_img = Image.new("RGB", (max(self.width, 300), max(self.height, 150)), (230, 230, 230))
            draw = ImageDraw.Draw(error_img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            message = "Error: Biblioteca vacía o no cargada."
            text_bbox = draw.textbbox((0, 0), message, font=font);
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((error_img.width - text_width) / 2, (error_img.height - text_height) / 2), message,
                      fill=(200, 0, 0), font=font)
            return error_img

        output_width = (self.width // self.block_size) * self.block_size
        output_height = (self.height // self.block_size) * self.block_size

        if output_width == 0 or output_height == 0 or self.block_size == 0:
            st.error("Tamaño de bloque inválido o demasiado grande. Ajusta el tamaño del bloque.")
            return self.target_image

        final_mosaic_image = Image.new("RGB", (output_width, output_height))

        sections = self._divide_into_sections(output_width, output_height)
        if not sections:
            st.warning("No se pudieron crear secciones para paralelizar (imagen/bloque muy pequeños).")
            if output_width > 0 and output_height > 0:
                target_avg = self._get_target_block_average_color(0, 0)
                item = self._find_best_match_tile_item(target_avg, None, allow_adjacent_repeats)
                if item: final_mosaic_image.paste(item['image_pil'], (0, 0))
            return final_mosaic_image

        results = [None] * len(sections)
        threads = []

        progress_bar, status_text_widget = None, None
        if show_progress_st_elements:
            progress_bar, status_text_widget = show_progress_st_elements
            status_text_widget.text("Iniciando creación del fotomosaico...")
            progress_bar.progress(0.0)

        for i, section_coords in enumerate(sections):
            thread = threading.Thread(
                target=self._process_section,
                args=(final_mosaic_image, section_coords, results, i, allow_adjacent_repeats)
            )
            threads.append(thread)
            thread.start()

        if progress_bar and status_text_widget:
            while any(thread.is_alive() for thread in threads):
                completed_count = sum(1 for r in results if r is True)
                progress_val = completed_count / len(sections) if len(sections) > 0 else 1.0
                progress_bar.progress(progress_val)
                status_text_widget.text(f"Procesando fotomosaico: {int(progress_val * 100)}%")
                time.sleep(0.1)

            progress_bar.progress(1.0)
            status_text_widget.text("¡Fotomosaico casi listo!")  # Mensaje intermedio

        for thread in threads:
            thread.join()

        if progress_bar and status_text_widget:  # Limpiar después de join
            status_text_widget.text("Fotomosaico completado.")
            time.sleep(0.5)
            status_text_widget.empty()
            progress_bar.empty()

        return final_mosaic_image


# --- Interfaz de Usuario de Streamlit ---
def create_ui():
    st.set_page_config(page_title="FotoMorsaicos", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    /* ... (CSS similar al anterior, puedes personalizarlo) ... */
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1, h2, h3 { color: #0072C6; } /* Un azul diferente */
    .stButton>button { border-radius: 5px; font-weight: bold; }
    .stFileUploader label, .stTextInput label, .stSelectbox label, .stSlider label, .stCheckbox label { font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # Inicializar estado de sesión
    if 'photomosaic_result' not in st.session_state: st.session_state.photomosaic_result = None
    if 'library_thumbnails_list' not in st.session_state: st.session_state.library_thumbnails_list = []
    if 'last_processed_lib_path' not in st.session_state: st.session_state.last_processed_lib_path = ""
    if 'last_processed_block_size' not in st.session_state: st.session_state.last_processed_block_size = 0
    if 'target_image_uploaded_name' not in st.session_state: st.session_state.target_image_uploaded_name = None
    if 'library_path_input_value' not in st.session_state: st.session_state.library_path_input_value = ""

    with st.sidebar:
        # Puedes poner aquí una imagen de "La Morsa" si la tienes.
        # st.image("path_to_your_morsa_logo.png", width=100)
        st.title("🖼️ FotoMorsaicos")
        st.markdown("---")

        uploaded_file = st.file_uploader("1. Imagen Objetivo:", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])
        library_path_input = st.text_input("2. Ruta a Biblioteca de Imágenes:",
                                           value=st.session_state.library_path_input_value)
        st.session_state.library_path_input_value = library_path_input  # Persistir

        st.markdown("---")
        st.subheader("Parámetros del Mosaico:")
        block_size_selected = st.select_slider("Tamaño del Bloque (px):", options=[8, 12, 16, 24, 32, 48, 64, 96],
                                               value=8)

        metric_options = {"Euclidiana": "euclidean", "Lineal (Windows)": "linear",
                          "Riemersma (Perceptual)": "riemersma"}
        selected_metric_key = st.selectbox("Métrica de Distancia:", options=list(metric_options.keys()), index=0)
        actual_metric_value = metric_options[selected_metric_key]

        st.markdown("---")
        st.subheader("Mejoras Opcionales:")
        apply_contrast = st.checkbox("Mejorar Contraste Imagen Objetivo")
        contrast_factor = 1.0
        if apply_contrast:
            contrast_factor = st.slider("Factor de Contraste:", 0.5, 3.0, 1.5, 0.1)

        apply_blending = st.checkbox("Aplicar Blending con Original")
        blend_alpha = 0.0
        if apply_blending:
            blend_alpha = st.slider("Factor de Blending (Alpha):", 0.0, 1.0, 0.25, 0.05,
                                    help="0.0 = Mosaico puro, 1.0 = Imagen original.")

        allow_adjacent_repeats_ui = st.checkbox("Permitir Repeticiones Adyacentes", value=True,
                                                help="Desmarcar para intentar que teselas idénticas no estén una al lado de la otra (horizontalmente).")

        st.markdown("---")
        st.caption("Proyecto final de Joel Miguel Maya Castrejón. FotoMorsaicos para la materia de Proceso Digital de Imágenes.")

    st.title("Generador de FotoMorsaicos")
    st.markdown("Crea fotomosaicos en base a un conjunto de imágenes.")
    st.markdown("---")

    if uploaded_file is not None:
        if st.session_state.target_image_uploaded_name != uploaded_file.name:  # Reset si cambia la imagen
            st.session_state.photomosaic_result = None
            st.session_state.target_image_uploaded_name = uploaded_file.name

        target_image_pil_original = Image.open(uploaded_file)

        # Aplicar mejora de contraste si está seleccionada
        target_image_to_process = target_image_pil_original
        if apply_contrast:
            enhancer = ImageEnhance.Contrast(target_image_pil_original)
            target_image_to_process = enhancer.enhance(contrast_factor)

        col_display1, col_display2 = st.columns(2)
        with col_display1:
            st.subheader("Imagen Objetivo")
            st.image(target_image_to_process, use_container_width=True,
                     caption=f"Entrada: {uploaded_file.name}{' (Contraste mejorado)' if apply_contrast else ''}")

        with col_display2:
            st.subheader("Fotomosaico Resultante")

            library_ready = False
            if library_path_input:
                if not os.path.isdir(library_path_input):
                    st.warning(f"La ruta '{library_path_input}' no es un directorio válido.")
                    st.session_state.library_thumbnails_list = []
                else:
                    # Usar library_path_input como parte de la "key" para el cache
                    thumbnails = get_processed_library_thumbnails(library_path_input, library_path_input,
                                                                  block_size_selected)
                    if thumbnails:
                        st.session_state.library_thumbnails_list = thumbnails
                        # No es necesario actualizar last_processed aquí, la función cacheada maneja la lógica de recarga.
                        if not st.session_state.get(f"success_msg_displayed_{library_path_input}_{block_size_selected}",
                                                    False):
                            st.success(
                                f"Biblioteca procesada: {len(thumbnails)} imágenes listas desde '{os.path.basename(library_path_input)}'.")
                            st.session_state[
                                f"success_msg_displayed_{library_path_input}_{block_size_selected}"] = True  # Evitar mensajes repetidos
                        library_ready = True
                    else:
                        st.session_state.library_thumbnails_list = []
                        if not st.session_state.get(f"error_msg_displayed_{library_path_input}", False):
                            st.error(f"No se pudieron cargar imágenes de la biblioteca en '{library_path_input}'.")
                            st.session_state[f"error_msg_displayed_{library_path_input}"] = True
            else:
                st.session_state.library_thumbnails_list = []  # Limpiar si no hay ruta

            if st.button("🚀 Crear Fotomosaico", type="primary", use_container_width=True, disabled=not library_ready):
                st.session_state.photomosaic_result = None  # Limpiar resultado anterior

                progress_bar_ph = st.empty()
                status_text_ph = st.empty()
                actual_prog_bar = progress_bar_ph.progress(0)  # Crear widgets reales
                actual_stat_text = status_text_ph.text("Preparando...")

                processor = PhotomosaicProcessor(target_image_to_process, block_size_selected, actual_metric_value)
                processor.library_thumbnails = st.session_state.library_thumbnails_list

                mosaic_image = processor.create_photomosaic(
                    allow_adjacent_repeats=allow_adjacent_repeats_ui,
                    show_progress_st_elements=(actual_prog_bar, actual_stat_text)
                )

                # Aplicar blending si está seleccionado
                if apply_blending and blend_alpha > 0.0:
                    actual_stat_text.text("Aplicando blending...")  # Actualizar estado
                    try:
                        # Redimensionar imagen original al tamaño del mosaico para el blend
                        original_resized_for_blend = target_image_pil_original.resize(mosaic_image.size,
                                                                                      Image.Resampling.LANCZOS)
                        final_image_for_display = Image.blend(mosaic_image, original_resized_for_blend, blend_alpha)
                        st.session_state.photomosaic_result = final_image_for_display
                        actual_stat_text.text("Blending aplicado. ¡Listo!")
                        time.sleep(0.5)
                    except Exception as e:
                        st.error(f"Error durante el blending: {e}")
                        st.session_state.photomosaic_result = mosaic_image  # Mostrar mosaico sin blend
                        actual_stat_text.text("Error en blending, mostrando mosaico base.")
                else:
                    st.session_state.photomosaic_result = mosaic_image

            if st.session_state.photomosaic_result is not None:
                current_result_img = st.session_state.photomosaic_result
                st.image(current_result_img, use_container_width=True,
                         caption=f"Mosaico: {block_size_selected}px, Métrica: {selected_metric_key}"
                                 f"{', Blend: ' + str(round(blend_alpha, 2)) if apply_blending and blend_alpha > 0 else ''}")
                buf = io.BytesIO()
                current_result_img.save(buf, format="PNG")
                st.download_button(label="⬇️ Descargar Fotomosaico (PNG)", data=buf.getvalue(),
                                   file_name=f"fotomosaico_{os.path.splitext(uploaded_file.name)[0]}_{block_size_selected}px.png",
                                   mime="image/png", use_container_width=True)
            elif not library_path_input:
                st.info("👈 Ingresa la ruta a tu biblioteca de imágenes en la barra lateral.")
            elif not library_ready and library_path_input:  # Ruta dada pero la biblioteca no cargó
                st.warning("Verifica la ruta de la biblioteca o su contenido. No se han cargado imágenes.")
            else:  # Biblioteca lista, esperando
                st.info("👍 ¡Todo listo! Haz clic en 'Crear Fotomosaico'.")

    else:
        st.info("👋 **¡Bienvenido!** Sube una imagen y configura la ruta de donde provienen tus imágenes.")


def main():
    try:
        create_ui()
    except Exception as e:
        st.error("Ocurrió un error mayor en la aplicación. Considera recargar la página.")
        st.exception(e)


if __name__ == "__main__":
    main()