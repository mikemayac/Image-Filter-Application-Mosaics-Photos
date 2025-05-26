# Filtro de Imagen Recursiva con Streamlit

### Joel Miguel Maya Castrejón │ mike.maya@ciencias.unam.mx │ 417112602

Este proyecto consiste en una aplicación web interactiva creada con **Python** y **Streamlit** que permite aplicar un filtro **recursivo** (o tipo fotomosaico fractal) a una imagen. La idea principal es **dividir** la imagen en bloques y **reemplazar** cada bloque por la propia imagen reescalada, ajustando los colores para que coincidan con el color promedio del bloque. Esto crea un efecto en el que la imagen está compuesta “infinitamente” de versiones en miniatura de sí misma (se suele realizar solo una o pocas iteraciones para no alargar el tiempo de cómputo).

## Requisitos

- Python 3.12 o superior.
- [Streamlit](https://docs.streamlit.io/) para el desarrollo de la interfaz.
- [Pillow](https://pillow.readthedocs.io/) (PIL) para leer y manipular las imágenes.
- [NumPy](https://numpy.org/) para operaciones de procesamiento vectorizado.
- [Threading](https://docs.python.org/3/library/threading.html) (parte de la librería estándar) para la ejecución en paralelo.

En el archivo **requirements.txt** están listadas las dependencias necesarias (Streamlit, Pillow y NumPy). Asegúrate de instalarlas antes de ejecutar la aplicación.

## Instalación

1. **Clona** o **descarga** este repositorio en tu máquina local.
2. Crea un **entorno virtual** (opcional, pero recomendado) e instálalo:
   ```bash
   python -m venv venv
   source venv/bin/activate        # En Linux/Mac
   # o en Windows: venv\Scripts\activate
   ```
3. Instala los paquetes necesarios:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución de la Aplicación

1. Dentro del entorno virtual y en la carpeta donde se encuentra el archivo principal (por ejemplo, `imagen_recursiva.py`), ejecuta:
   ```bash
   streamlit run imagen_recursiva.py
   ```
2. Automáticamente se abrirá tu navegador mostrando la interfaz de la aplicación. Si no se abre, puedes copiar la URL que aparece en la terminal y pegarla en tu navegador.

## Uso de la Aplicación

1. **Sube una imagen** en la barra lateral (sidebar). Acepta formatos `JPG`, `JPEG`, `PNG`, entre otros. En caso de que la imagen esté en formato RGBA, se convierte a RGB.
2. **Selecciona** el tamaño de bloque en píxeles (por ejemplo, 10 px) y el nivel de calidad del procesamiento (baja, normal, alta o ultra).
3. Observa cómo se muestra la **imagen original** en la columna izquierda y la **imagen resultante** (recursiva) en la columna derecha.
4. Puedes **descargar** la imagen procesada usando el botón de descarga que aparece en la interfaz.

### Explicación del Filtro Recursivo

- La imagen se **divide** en bloques de tamaño fijo.
- Para cada bloque, se calcula el **color promedio** utilizando operaciones vectorizadas de NumPy.
- Se crea una versión en miniatura (tile) de la imagen completa y se **recolorea** para ajustar sus tonos al color promedio del bloque original.
- El tile recoloreado se **pega** en la posición correspondiente de la imagen de salida.
- Se utiliza **procesamiento paralelo** (mediante hilos) para acelerar la aplicación del filtro en diferentes secciones de la imagen.

## Estructura del Proyecto

```
├── imagen_recursiva.py    # Código principal de la aplicación que incluye la clase RecursiveImageProcessor y la interfaz de usuario con Streamlit
│── .streamlit/            # Carpeta de configuración de Streamlit
│    └── config.toml       
├── README.md              # Archivo de documentación
├── requirements.txt       # Dependencias del proyecto (Streamlit, Pillow, NumPy)
└── venv/                  # Entorno virtual (opcional)
```

## Detalles Técnicos

- **Clase RecursiveImageProcessor:**  
  Implementa el procesamiento de imágenes dividiéndolas en bloques, calculando el color promedio de cada bloque y generando un efecto recursivo mediante la sustitución de cada bloque por una versión en miniatura de la imagen original.  
  Además, utiliza técnicas de **caché** (con `@lru_cache`) para optimizar el cálculo de promedios de color y **procesamiento en paralelo** (con `threading`) para mejorar el rendimiento.

- **Interfaz con Streamlit:**  
  La interfaz permite al usuario cargar imágenes, ajustar parámetros como el tamaño de bloque y la calidad, y ver en tiempo real el progreso y resultado del filtro recursivo. También ofrece la opción de descargar la imagen procesada.

- **Procesamiento Eficiente:**  
  Se aprovecha la biblioteca NumPy para realizar operaciones vectorizadas y se implementa el procesamiento paralelo, lo que permite obtener resultados de alta calidad sin tiempos de espera excesivos.