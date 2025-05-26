# FotoMorsaicos Pro: Generador de Fotomosaicos con Streamlit

### Joel Miguel Maya Castrejón | mike.maya@ciencias.unam.mx | 417112602

Este proyecto consiste en una aplicación web interactiva creada con **Python** que permite generar 
**fotomosaicos**. La idea principal es tomar una imagen objetivo, dividirla en una cuadrícula de pequeñas 
regiones (bloques o teselas), y para cada región, sustituirla con una imagen específica de una biblioteca de imágenes. 
La selección de la imagen de la biblioteca se basa en la similitud de color con la región de la imagen objetivo, 
utilizando diversos criterios o métricas de distancia.

## Requisitos

- Python 3.9 o superior (recomendado 3.10+).
- **Streamlit**: Para la interfaz web interactiva.
- **Pillow (PIL)**: Para la carga, manipulación y procesamiento de imágenes.
- **NumPy**: Para cálculos numéricos eficientes y operaciones vectorizadas con los datos de las imágenes.

Se tiene un archivo `requirements.txt` con:
- streamlit~=1.41.1
- pillow~=11.1.0

## Instalación

1.  **Clona** o **descarga** el repositorio en tu máquina local.
2.  Crea un **entorno virtual** (opcional, pero altamente recomendado):
    ```bash
    python -m venv venv
    # En Linux/Mac:
    source venv/bin/activate
    # En Windows:
    # venv\Scripts\activate
    ```
3.  Instala los paquetes necesarios. Si tienes un archivo `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Ejecución de la Aplicación

1.  Navega en tu terminal a la carpeta donde guardaste el archivo principal (ej., `fotomosaicos.py`).
2.  Si estás usando un entorno virtual, asegúrate de que esté activado.
3.  Ejecuta la aplicación Streamlit:
    ```bash
    streamlit run fotomosaicos.py
    ```
4.  El navegador web debería abrirse automáticamente mostrando la interfaz de la aplicación. Si no, copia la URL local (usualmente `http://localhost:8501`) que aparece en la terminal y pégala en tu navegador.

## Uso de la Aplicación

1.  **Sube tu Imagen Objetivo**: Utiliza el cargador de archivos en la barra lateral para seleccionar la imagen principal que deseas recrear como un fotomosaico.
2.  **Especifica la Ruta a tu Biblioteca de Imágenes**: Ingresa la ruta completa a la carpeta en tu computadora que contiene las imágenes pequeñas (teselas) que se usarán para construir el mosaico.
3.  **Ajusta los Parámetros del Mosaico**:
    * **Tamaño del Bloque**: Define la resolución del fotomosaico. Bloques más pequeños significan más detalle pero un procesamiento más largo y la necesidad de una biblioteca de imágenes más grande y variada.
    * **Métrica de Distancia**: Elige el algoritmo para comparar el color de una región de la imagen objetivo con las imágenes de la biblioteca (por ejemplo, Euclidiana, Lineal, Riemersma).
4.  **Configura Mejoras Opcionales**:
    * **Mejora de Contraste**: Puedes optar por realzar el contraste de la imagen objetivo antes del procesamiento.
    * **Blending**: Permite mezclar el fotomosaico final con la imagen objetivo original para suavizar el efecto o hacerlo más fiel a los colores originales.
    * **Permitir Repeticiones Adyacentes**: Controla si la misma imagen de la biblioteca puede usarse para teselas horizontalmente adyacentes.
5.  **Crea y Descarga**:
    * Haz clic en el botón "Crear Fotomosaico".
    * Observa la imagen original (con contraste aplicado si se seleccionó) y el fotomosaico resultante.
    * Puedes descargar la imagen del fotomosaico procesada (generalmente en formato PNG).

### Explicación del Proceso de Fotomosaico

1.  **Carga y Preprocesamiento de la Biblioteca**:
    * Las imágenes de la carpeta especificada son cargadas.
    * Cada imagen de la biblioteca se redimensiona al "Tamaño del Bloque" seleccionado.
    * Se calcula y almacena el color promedio de cada una de estas miniaturas redimensionadas. Este proceso se cachea para acelerar cargas futuras si la ruta y el tamaño del bloque no cambian.
2.  **Procesamiento de la Imagen Objetivo**:
    * La imagen objetivo se divide conceptualmente en una cuadrícula, donde cada celda tiene el "Tamaño del Bloque". 
    * Para cada bloque/región de la imagen objetivo:
        * Se calcula su color promedio.
        * Usando la **Métrica de Distancia** seleccionada, se compara este color promedio con los colores promedio de todas las miniaturas preprocesadas de la biblioteca. 
        * La miniatura de la biblioteca cuyo color es el "más cercano" (menor distancia) se selecciona para ese bloque.
        * La miniatura seleccionada se pega en la posición correspondiente de la imagen de salida.
3.  **Paralelización**: El proceso de analizar bloques y seleccionar teselas se realiza en paralelo para diferentes secciones de la imagen, utilizando múltiples hilos (`threading`) para acelerar la creación del fotomosaico.
4.  **Mejoras**: Si se seleccionan, se aplica el realce de contraste a la imagen objetivo antes del procesamiento, y/o el blending al resultado final.