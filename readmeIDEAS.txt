
IDEAS
Para leer y utilizar las anotaciones
Con Python + OpenCV o Pillow, se pueden cargar las coordenadas y dibujar la máscara del defecto.

Grupo de IA (Inteligencia Artificial) Responsabilidad principal: 
Entrenamiento del modelo para la detección de anomalías.
Tareas a realizar: 
	• Preparación del dataset: 
		○ Entrenamiento de un modelo autoencoder: El modelo aprenderá las características de las imágenes sin defectos (normalidad) y, luego, podrá identificar las anomalías observando los errores de reconstrucción. 
		○ Preprocesamiento de imágenes: 
			- Redimensionar las imágenes a un tamaño uniforme (por ejemplo, 500x500 píxeles). 
			- Normalizar las imágenes (valores de píxeles entre 0 y 1). • Entrenamiento del modelo: 
	○ Usar TensorFlow o PyTorch para construir un autoencoder: 
		- Capa de codificación: Codifica las características importantes de la imagen. 
		- Capa de decodificación: Reconstruye la imagen desde la codificación, y cuanto mayor sea el error de reconstrucción, mayor es la probabilidad de que sea una anomalía. 
	○ Entrenar el modelo con las imágenes sin defectos (156 imágenes) para que aprenda la normalidad.
• Evaluación del modelo: 
	-Ajustar los umbrales de error de reconstrucción para detectar las anomalías correctamente (reducir los falsos positivos).