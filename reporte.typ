#set document(
  title: [Buscando simetrías en funciones mediante redes neuronales],
  author: ("Darío Penagos", "Daniel")
)

#title()
*Autores:* Daniel, Darío Penagos


= Introducción

En la física, un problema común es encontrar simetrías en datos correspondientes a algún fenómeno. Desafortunadamente, a menudo los datos disponibles no son suficientes para determinar si un fenómeno exhibe o no una simetría dada.

Una forma de resolver éste problema sería utilizando un ajuste de curvas: Dado un conjunto de datos del que queremos verificar la existencia de simetrías, podemos ajustar una función $hat(f)$ a dichos datos, y verificar si $hat(f)$ tiene o no las simetrías consideradas. Luego, si $hat(f)$ exhibe una simetría dada, es probable que la simetría también esté presente en la función original.

En @article, S. Udrescu y M. Tegmark desarrollan un algoritmo de regresión simbólica que consiste de varios "módulos" donde uno de dichos módulos consiste en utilizar una red neuronal para aproximar los datos considerados e intentar encontrar una simetría en la red neuronal, la cual luego se utiliza restringir el rango de fórmulas que se pueden utilizar para aproximar los datos. En el presente texto, buscamos reproducir sus resultados. Es decir, entrenaremos redes neuronales con la misma arquitectura y datos para verificar si dichas redes neuronales exhiben alguna simetría que podría ser útil para obtener un mejor entendimiento del fenómeno físico considerado.

= Metodología

== Arquitectura
La arquitectura y método de entrenamiento se encuentran en el archivo `train.py`. Todas las redes neuronales utilizan 3 capas ocultas completamente conexas con dimensiones $128 times 128$, $128 times 64$, $64 times 64$. Con función de activación $tanh$. Como optimizador, se utilizó `torch.optim.Adam`.

== Datos
Los datos usados fueron generados sintéticamente de un conjunto de ecuaciones sacadas de @lectures. Se determinaron 100.000 puntos arbitrarios en el dominio de la función considerada, y se computó el valor de la función para cada uno de éstos valores arbitrarios. Todas las funciónes son de la forma $RR^n->RR$. Una descripción más detallada de éstas funciones se encuentra en el archivo `FeynmanEquations.csv`.

== Algoritmo


#columns(
  2
)[
El optimizador utilizado es `torch.optim.Adam` con un `batch_size` de 2048.

El valor de `lr` inicia en `1e-2`. Luego, se entrena una red neuronal por 1.000 épocas. Al final de éste proceso, el valor de `lr` se divide por 10. Dicho proceso se repite 4 veces. Si el algoritmo detecta que la red neuronal ha dejado de aprender con el valor de `lr` actual, símplemente interrumpe el proceso y sigue entrenando con el `lr` dividido por 10, como si hubiesen pasado las 1.000 épocas. En pseudocódigo:

#colbreak()

```python
for _ in range(4):
  check_loss = 10_000
  for epoch in range(1000):
    for (X,Y) in dataloader:
      optimizer.zero_grad()
      err = loss(model(X),Y)
      err.backward()
      optimizer.step()
    if epoch%20==0 and epoch>0:
      if check_loss<err:
        break
      else: check_loss = err
```
]

== Reconocimiento simetrías

Comenzamos revisando manualmente las funciones y anotando cada una de las simetrías que cumplen en el archivo `Invariante_por_Funcion.csv`.














#bibliography("sources.yml")