# Tracker C++

Seguimiento de objetos a partir de detecciones por frame. Requiere un compilador
C++23, Make y Eigen 3 (por defecto en `/usr/include/eigen3`).

```sh
make
make run
make test
```

## Estructura

- `include/Tracker.hpp`: interfaz abstracta con `track()` y `reset()`, con destructor virtual.
- `include/ByteTracker.hpp`, `src/ByteTracker.cpp`: configuración y asociación en dos etapas.
- `include/Detection.hpp`: caja `(x, y, width, height)` en píxeles, confianza `[0, 1]` e IoU.
- `include/Prediction.hpp`: resultado por valor, con ID estable y caja estimada.
- `include/Track.hpp`, `src/Track.cpp`: trayectoria, Kalman y estados `Tentative`, `Tracked`, `Lost`, `Removed`.
- `include/KalmanFilter.hpp`: filtro existente de velocidad constante `[cx, cy, vx, vy]`.
- `include/HungarianAlgorithm.hpp`, `src/HungarianAlgorithm.cpp`: asignación global rectangular con umbral y filas sin pareja.
- `src/main.cpp`: ejemplo de uso a través de la interfaz.
- `tests/`: pruebas del filtro, asignación y ciclo de vida de tracks.

## Uso y extensión

```cpp
#include "ByteTracker.hpp"
#include <memory>

std::unique_ptr<Tracker> tracker = std::make_unique<ByteTracker>();
auto predictions = tracker->track({Detection(10, 20, 40, 60, 0.9)});
for (const auto &prediction : predictions) {
    auto id = prediction.id();
    const auto &box = prediction.box();
    // Consumir id y box.
}
```

Para añadir otro tracker, heredar de `Tracker` e implementar `track()` y `reset()`.
La interfaz pública no depende de Kalman ni del algoritmo húngaro; cada
implementación puede elegir su modelo de movimiento y asociación.
`Track` y `HungarianAlgorithm` son componentes separados que se pueden reutilizar.

Llamar a `track()` una vez por frame, incluso con un vector vacío. El paso temporal
de Kalman es un frame. El resultado contiene solo tracks confirmados observados en
ese frame, con la caja corregida por Kalman; los perdidos se conservan internamente
hasta `maxLostFrames` frames ausentes. Pueden recuperarse en el siguiente frame si
no han excedido ese número de ausencias. `reset()` inicia una secuencia nueva y
reinicia los IDs a 1. Los IDs son locales a cada instancia y secuencia.

## ByteTrack implementado

1. Predice el centro de cada track con Kalman.
2. Asocia tracks confirmados y perdidos con detecciones de alta confianza mediante
   coste `1 - IoU * confianza` y asignación húngara.
3. Asocia tracks activos todavía sin pareja con detecciones de baja confianza
   mediante coste `1 - IoU`. Las detecciones débiles no crean ni reactivan tracks.
4. Confirma tracks tentativos con las detecciones fuertes restantes. Elimina los
   tentativos sin pareja y crea nuevas trayectorias con `newTrackConfidence`.
5. Conserva los tracks perdidos durante el buffer configurado y elimina los caducados.

Los nacimientos del primer frame se confirman inmediatamente; los posteriores
necesitan otra detección fuerte en el frame siguiente. Los umbrales de asociación
son costes máximos: un valor menor es más estricto. La asignación maximiza primero
el número de parejas válidas y después minimiza su coste total.

Esta implementación adapta la asociación en dos etapas de
[ByteTrack](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py).
No es una reproducción exacta del tracker de referencia: reutiliza el Kalman 2D
existente y mantiene el último ancho y alto observado, en lugar de filtrar también
la relación de aspecto y la altura. No incluye la supresión de duplicados entre
tracks activos y perdidos. Las detecciones deben llegar ya depuradas (por ejemplo,
con NMS); esta versión no distingue clases de objetos.

## Optimización con Eigen

`AssociationCost.hpp` y `src/AssociationCost.cpp` construyen los costes IoU con
operaciones por coeficiente sobre arrays Eigen. Las coordenadas y áreas de las
detecciones se preparan una vez por asociación; cada caja de track se obtiene una
vez. `CostMatrix` usa almacenamiento contiguo por filas, acorde al recorrido del
algoritmo húngaro. `HungarianAlgorithm::solve()` recibe
`Eigen::Ref<const CostMatrix>` para evitar copiar esa matriz. Los vectores de
índices y el control de estados siguen usando contenedores estándar.

Kalman usa bloques de tamaño fijo para la observación `H = [I2, 0]`, conserva la
resolución LDLT y la actualización de covarianza de Joseph. Los productos que
escriben sobre la covarianza utilizan intermedios independientes antes de aplicar
`noalias()`, siguiendo las [reglas de aliasing de Eigen](https://libeigen.gitlab.io/eigen/docs-nightly/group__TopicAliasing.html).

El Makefile activa `-O2` por defecto. Se puede sobrescribir `CXXFLAGS` para depurar.
Para comparar la construcción escalar de costes con la versión Eigen:

```sh
make benchmark
```

El benchmark incluye la preparación y reserva de las matrices, con 1.000
repeticiones para tamaños 16, 64 y 256. No mide el tracker completo ni garantiza
una mejora concreta en otro hardware. Las pruebas contrastan los costes Eigen
con IoU escalar y la asignación húngara con un óptimo calculado exhaustivamente.
