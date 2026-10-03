# Lab 5: Key Terms and What the Professor Wanted You to Learn

It's about understanding how to evaluate whether two images show the same object or a similar visual region, and how to decide whether a match is useful or false.

## 1. What is the overall goal? 

To learn to build a reproducible computer vision workflow in which:

a. load inspection images,
b. detect distinctive visual points,
c. describe those points with descriptors,
d. compare two images to see which regions match,
e. and then check whether those matches are actually plausible or false.

## 2. Important terms   

### 2.1 Manifest

`data/manifest.csv` is a metadata file that describes each image.

It includes columns such as:

- `image`: file name    
- `label`: label or class associated with the image  
- `split`: indicates whether the image belongs to train, validation, or test
- `condition`: describes the visual state of the image, for example  `reference` o `dim`
- `width` y `height`: image size

### 2.2 Split: train, validation y test


- train: images used to train or tune the analysis method
- validation: images used to evaluate and compare strategies
- test: images reserved for a final test, not used to choose the solution

### 2.3 Condition

`condition` indicates the visual state of the image. In this case:

- `reference`: the image looks normal, with the expected lighting and appearance
- `dim`: the image is darker or has lower contrast

Visual correspondence algorithms don't always perform the same way when lighting changes.

### 2.4 Grayscale image

Before detecting keypoints, the image is converted to black and white with `cv2.cvtColor()`.

Why?

- It reduces complexity: the three color channels are no longer considered
- It makes processing more efficient
- It makes it easier to extract edges, textures, and visual structures
- Many detectors, such as ORB, work better on grayscale images


### 2.5 Keypoints 

Keypoints are visually distinctive locations in the image, such as corners, edges, textures, marks, or patterns.

These points matter because they let you compare one image with another without comparing every pixel.

The professor wants you to understand that an image isn't compared as a whole; important regions are compared.

### 2.6 ORB

ORB stands for Oriented FAST and Rotated BRIEF. It is a widely used feature detector and descriptor because it: is fast, is relatively lightweight, is used in visual matching applications, and produces binary descriptors.

In this assignment, ORB is the main tool for detecting points of interest and describing them.

### 2.7 nfeatures

`nfeatures` is the maximum number of keypoints ORB will try to detect.

Example:

orb = cv2.ORB_create(nfeatures=500) - This means: "try to extract up to 500 relevant keypoints."

Why is it important?

- If it is set too low, there may not be enough points to compare.
- If it is set too high, you may detect points that are not very useful or are repetitive.
- In small images (96x96), the number does matter.


### 2.8 Descriptores

A descriptor is a numerical signature that represents the local appearance around a keypoint. Instead of storing the whole pixel region, a compact representation is stored.

In ORB, the descriptor is binary: that is, it is made up of 0s and 1s. This makes sense because ORB compares local intensity in a compact and efficient way.

### 2.9 Hamming distance

Hamming distance measures how many positions differ between two binary strings.

If two descriptors are very similar, their Hamming distance will be small.

Example:

- descriptor A: 10101100
- descriptor B: 10100100

Here there is only one difference, so the distance is 1.

With ORB, `cv2.NORM_HAMMING` is used because the descriptors are binary. Euclidean distance is not the right choice here.

### 2.10 BFMatcher

`cv2.BFMatcher` s a descriptor matcher.

It works by comparing every descriptor in one image with every descriptor in the other, and then looking for matches based on distance.

It is a simple, correct, and useful method for understanding the basic process before applying more sophisticated filters.

### 2.11 Matches

A match is a correspondence between two keypoints: a point in image A and a point in image B that the algorithm considers similar.

Each match has:

- queryIdx: index of the point in the first image
- trainIdx: index of the point in the second image
- distance: how similar those descriptors arequeryIdx: 

### 2.12 Minimum, median, and maximum distance

When matches are computed, they are sorted by distance.

minimum: best match (closest)
median: central value of the set
maximum: worst or most distant match

These statistics help describe how strong or scattered the correspondence is.

### 2.13 Visualizing keypoints and matches

Images are saved showing:

keypoints highlighted in each image,
lines drawn between matches in the comparison of the two images.

This allows for human review. Visual interpretation is key.

It isn't enough to say "there were 18 matches"; you have to look at whether those lines connect relevant parts of the image or point to noise.

### 2.14 Cross-check matching

Many matches can be ambiguous or false. A common filter is the cross-check.

The idea: look for matches from A to B, then look for matches from B to A, and keep only the matches that are consistent in both directions.

It is a way to reduce false positives.

### 2.15 Lowe ratio test

This filter compares the best match with the second-best match. If the best match is not clearly better than the second, it is discarded.

It is a widely used criterion in SIFT/ORB-style matching.

### 2.16 Falso positivo / match dudoso

Not every match with a low distance is correct.

A visual false positive can be:

- a match on repeated edges, on text, on repeated patterns,
or on similar background areas.

Validation is not automatic; it requires interpretation and judgment.

## 3. ¿Qué quiso el profesor que aprendieras?

### 3.1 Reproducibility

The analysis must be repeatable with the same images, the same configurations, and the same steps. This is very important in science, engineering, and evaluation.

### 3.2 Evaluación basada en evidencia

No se acepta decir "parece bien" sin guardar imágenes y resultados. La evidencia debe dejarse documentada.

### 3.3 Entender el pipeline de correspondencia

La secuencia típica es:

1. Cargar imagen
2. Convertir a gris
3. Detectar keypoints
4. Crear descriptores
5. Match descriptors
6. Filtrar matches
7. Revisar visualmente
8. Concluir si la coincidencia es útil

### 3.4 Interpretar el resultado con criterio

- un match visualmente plausible,
- un match débil,
- un match sospechoso,
- y un match que parece correcto solo por similitud numérica.

### 3.5 Comprender que los datos importan

En esta tarea, las imágenes son pequeñas y sintéticas. Por eso, muchos patrones se repiten. Esto te enseña que incluso con un algoritmo correcto, hay ruido, ambigüedad y patrones engañosos.

### 3.6 Relación entre teoría y práctica

Es un ejemplo concreto de cómo la visión por computadora conecta:

- matemáticas (distancias, descriptores),
- procesamiento de imágenes,
- visión geométrica y de correspondencia,
- y validación experimental.

## 4. En una frase: la lección principal

construir un proceso que convierta información visual en decisiones explicables, medibles y verificables. Una coincidencia en una imagen no es necesariamente correcta solo porque el algoritmo la encontró. Hay que evaluarla con evidencia, criterio y revisión visual.

## 5. Resumen corto para recordar

- ORB detecta puntos relevantes.
- Los descriptores describen esas regiones.
- BFMatcher compara descriptores.
- Hamming distance mide similitud entre descriptores binarios.
- Los matches necesitan filtros y revisión visual.
- El trabajo científico en visión requiere evidencia, no solo resultados numéricos.
