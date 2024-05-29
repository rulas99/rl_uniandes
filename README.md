## Aprendizaje por Refuerzo Adaptativo: una aproximación al aprendizaje continuo
### Monitoría de investigación, 2024 - 1
### Raúl de la Rosa

---

## Descripción del Proyecto

Este proyecto aborda el aprendizaje por refuerzo, un paradigma de inteligencia artificial donde un agente aprende a través de la experiencia para resolver problemas, maximizando recompensas mediante la toma de decisiones óptimas. Se explora particularmente la capacidad de los agentes para adaptarse a entornos con recompensas dinámicas utilizando la aplicación clásica "Grid World".

## Contenido del Repositorio

### Archivos Principales

- **value_function.ipynb**: Implementación del algoritmo de evaluación de valores en un Grid World estático.
- **q_learning.ipynb**: Implementación del algoritmo Q-learning tradicional en un Grid World estático.
- **q_learning_cl.ipynb**: Implementación y análisis del comportamiento de agentes adaptativos y tradicionales en entornos dinámicos.
- **q_learning_cl_tests.ipynb**: Experimentos sobre cambios en recompensas del entorno y cómo el agente reacciona ante distintos escenarios.

### Carpetas

- **simulations**: Animaciones ilustrativas del comportamiento del agente en diversos experimentos.

## Actividades Realizadas

1. **Familiarización con la Teoría del Aprendizaje por Refuerzo**: Se estudiaron las funciones de recompensa y los algoritmos de predicción y control.
2. **Implementación en Python**: Configuración de agentes en un Grid World utilizando Q-learning.
3. **Experimentos con Recompensas Dinámicas**: Diseño de entornos cambiantes y análisis de la capacidad de los agentes para readaptarse.

## Descripción de los Experimentos

### Experimento Unidimensional

- Un agente se mueve a la izquierda o derecha para alcanzar un estado objetivo.
- Comportamiento del agente con Q-learning tradicional en un entorno con recompensas invertidas.

### Experimento Bidimensional

- Comparación entre agentes adaptativos y tradicionales en un Grid World bidimensional.
- Configuración del Grid World bidimensional con estados objetivo dinámicos.

## Implementación del Agente Adaptativo

Se desarrolló un agente capaz de percibir cambios abruptos en las recompensas, detonando una señal de "olvido" para reiniciar la exploración y adaptación. Esto se logra llevando un registro de las últimas recompensas y evaluando cambios significativos.

### Mejora en la Programación del Agente

Se introdujeron mejoras para que el agente pueda identificar cambios en el entorno y reajustar su comportamiento de manera eficiente, logrando una rápida convergencia a nuevas políticas cuando las recompensas cambian.

## Resultados

- Agente adaptativo que detecta cambios y activa la señal de "olvido".
- Agente adaptativo en un Grid World bidimensional, mostrando coherencia con los resultados unidimensionales.

## Conclusiones

El proyecto demuestra cómo un enfoque adaptativo puede mejorar significativamente la capacidad de un agente para aprender y reajustarse en entornos con recompensas dinámicas. Las implementaciones realizadas proporcionan una base sólida para futuros trabajos en el manejo eficiente de la memoria y la optimización del aprendizaje continuo en agentes de inteligencia artificial.

Para más detalles, consulta los archivos y las animaciones en el repositorio.
