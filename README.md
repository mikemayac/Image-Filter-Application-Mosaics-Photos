# Aplicación de Filtro de Mosaico con Círculos y Estrellas

### Recreación de Efectos Artísticos Estilo "Pointillismo"

Esta aplicación web, construida con **Python** y **Streamlit**, permite aplicar un **filtro de mosaico** que, en lugar de bloques cuadrados, utiliza **figuras** (círculos o estrellas) para recrear tu imagen con un estilo que recuerda a técnicas de pointillismo o mosaico artístico.

El proceso general consiste en:
1. **Dividir la imagen en celdas** de tamaño fijo.
2. **Calcular el color promedio** de cada celda.
3. **Dibujar una figura** (círculo o estrella) con el color promedio en la posición correspondiente.

El resultado es un efecto visual muy llamativo, que puede modificarse ajustando el tamaño de las celdas o incluso la forma de las figuras, dando lugar a patrones variados y originales.

---

## Requisitos

- Python 3.8 o superior
- [Streamlit](https://docs.streamlit.io/) para la creación de la interfaz web.
- [Pillow](https://pillow.readthedocs.io/) (PIL) para la manipulación de imágenes.
- Math (librería estándar de Python) para cálculos de coordenadas de estrellas.

En el archivo **requirements.txt** se listan las dependencias necesarias. Asegúrate de instalarlas antes de ejecutar la aplicación.

---

## Instalación

1. [**Clona** este repositorio](https://github.com/mikemayac/Image-Filter-Application-Mosaics) en tu máquina local.
2. Crea y activa un **entorno virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/Mac
   # En Windows: venv\Scripts\activate
   ```
3. Instala los paquetes necesarios:
   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución de la Aplicación

1. Dentro del entorno virtual, ubícate en la carpeta donde se encuentra el archivo principal (por ejemplo, `mosaicos.py`).
2. Ejecuta:
   ```bash
   streamlit run mosaicos.py
   ```
3. Tu navegador abrirá la interfaz de la aplicación.  
   Si no se abre automáticamente, copia la URL que aparece en la terminal y pégala en tu navegador.

---

## Uso de la Aplicación

1. **Sube una imagen** en la barra lateral, en formatos `JPG`, `JPEG` o `PNG`.
2. Ajusta los parámetros del filtro:
   - **Tamaño de cada mosaico (cell_size)**: Controla cuán grande es la celda (y por ende el tamaño de los círculos/estrellas).
   - **Forma del mosaico**: Selecciona entre *Circulos* o *Estrellas*.
3. **Observa** cómo se muestra la **imagen original** en una columna y la **imagen con el filtro** en la otra.
4. **Descarga** la imagen filtrada pulsando el botón de descarga situado sobre la imagen resultante.

---

## Algoritmo Implementado

1. **División en celdas**  
   El programa recorre la imagen en bloques de tamaño fijo (`cell_size`). Para cada bloque, obtendrá el color promedio.
   
2. **Cálculo de color promedio**  
   Se suman los valores de cada píxel (R, G, B) dentro de la celda y se dividen entre la cantidad de píxeles de la misma.

3. **Dibujo de figuras**  
   - **Círculos**: Se calcula el centro de cada bloque y se dibuja un círculo con el color promedio.
   - **Estrellas**: Se genera una lista de coordenadas formando la estrella (cálculo trigonométrico) y se dibuja un polígono con el color promedio.

Estas operaciones se realizan sobre una **nueva imagen** en blanco de las mismas dimensiones, donde se agregan las figuras una a una, formando finalmente el mosaico.

---

## Estructura del Proyecto

```bash
.
├── mosaicos.py                # Código principal de la aplicación (filtro mosaico)
├── .streamlit/                # Configuraciones extra de Streamlit
│    └── config.toml           
├── README.md                  # Archivo de documentación (este archivo)
├── requirements.txt           # Dependencias del proyecto
└── venv/                      # Entorno virtual (puede variar según tu instalación)
```

---