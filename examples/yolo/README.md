# Vídeo de tráfico con YOLO y ByteTracker C++

Lee cada frame con OpenCV, ejecuta YOLO y pasa las detecciones al tracker de
este repositorio. Muestra el vídeo con las cajas e IDs de los objetos seguidos.

Desde la raíz del repositorio:

```sh
uv sync --package yolo
uv run --package yolo main
```

Necesitas Python 3.14+, compilador C++23 y Eigen 3, como el proyecto principal.
El ejemplo lee `src/traffic.mp4`, junto a `main.py`. Puedes cambiar
`VIDEO_URL` para utilizar otro archivo. La primera ejecución descarga los pesos
`yolo26n.pt`. Pulsa **Q** para salir de la ventana.

Desde `examples/yolo/` puedes ejecutar `uv run main`. El punto de entrada es
`src.main:main`, que llama a la función `main()` de `src/main.py`.
No hay `__init__.py`; se configura el [paquete namespace de uv](https://docs.astral.sh/uv/concepts/build-backend/#namespace-packages)
con `namespace = true`.

El formato de entrada se configura antes de instanciar el tracker:

```python
tracker_config = tracker.ByteTrackerConfig()
tracker_config.inputFormat = tracker.BoxFormat.CXCYWH
byte_tracker = tracker.ByteTracker(tracker_config)

detections[:, :4] = boxes.xywh.cpu().numpy()
```

`CXCYWH` acepta centro x, centro y, ancho y alto de YOLO, en píxeles, sin restar
manualmente la mitad de las dimensiones. Para usar `boxes.xyxy`, configura
`tracker.BoxFormat.XYXY`. `tracker.BoxFormat.TLWH` es el formato por defecto:
esquina superior izquierda, ancho y alto.

Las columnas restantes del array `float32` `(N, 6)` son confianza y clase.
Se llama a `update()` en cada frame, incluso con un array vacío `(0, 6)`.
La salida siempre es `[x, y, ancho, alto, id, confianza, clase]` en `TLWH`,
lista para dibujar desde `(x, y)` hasta `(x + ancho, y + alto)` con OpenCV.
