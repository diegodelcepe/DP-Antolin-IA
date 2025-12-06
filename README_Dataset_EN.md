# DP-Antolin-IA Project - Complete Documentation

## General Dataset Documentation — DP-Antolin-IA

This document describes the **visual dataset** used in the automatic inspection challenge for plastic parts in the **DP-Antolin-IA** project. It is the fundamental resource that enables the training and validation of the **anomaly detection** model through computer vision (*PatchCore*).

---

## 1. Purpose of the Dataset

The dataset is used to **learn normality** and detect parts that deviate from the expected pattern.

*Benefits:*

* Reduces **material waste** in manufacturing
* Decreases **rejections** and rework
* Improves inspection **quality**
* Aligns with **industrial sustainability** approach

---

## 2. Dataset Content

| Image Type     | Quantity | Description                              |
| -------------- | -------- | ---------------------------------------- |
| No Defect      | 156      | Normality examples                       |
| With Defect    | 45       | Anomalous regions marked with polygons   |
| **TOTAL**      | **201**  | Real production images                   |

*Technical characteristics:*

* Average resolution: **Approx. 500x500 px**
* Captured with **fixed camera**
* Slight positional variation due to robotic handling

Defect labeling:

* Annotations in **JSON** per image with **polygons**
* **LabelMe** standard
* Same name for image and annotation

### Visual Examples of the Dataset

| Normal | Defect |
|--------|---------|
| <img src="Dataset/dataset_gua_crops/cropped_images/normales/img_2024-12-03_08.47.57_11150774_cam_H2674069.png" width="200"> | <img src="Dataset/dataset_gua_crops/cropped_images/defectuosas/img_2024-12-03_10.41.24_11152208_cam_H2674069.png" width="200"> |

---

## 3. Location and Organization in the Project
```
Dataset/
├─ dataset_gua_crops/cropped_images # Images + JSON annotations
└─ dataset_gua_crops.zip # Compressed version
```

Usage within the system:

* Training of PatchCore's **memory bank**
* Direct input to the `POST /predict` endpoint (backend)
* Visualization in the inspection **frontend**

---

## 4. Relationship with PatchCore (Backend)

PatchCore uses the dataset as follows:
1. Learns features from **normal** images
2. Builds a **memory bank** -> normality embeddings
3. If a new image **does not resemble** -> it's anomalous

### One-Time Creation of the "Brain" (Memory Bank)

To make the system fast and efficient, all the **heavy lifting was done once** at the beginning (in a development notebook). It was at that moment when the data was separated and key information from the images was extracted.

The result of that process is a single fixed file: **`memory_bank_core.npz`**.

**How does it work in the final app?**
The application **does not waste time learning again** or reorganizing photos. It simply loads that ready-made `.npz` file and uses it as a reference to instantly judge whether new parts are good or bad.

The backend returns:

* Anomaly **score**
* **Heatmap**
* Detected **polygons**
* *(Optional)* overlays

> This allows detection of **previously unknown defects** that were not previously labeled.

---

## 5. Impact on Sustainability

Thanks to this dataset, it is possible to:

* Detect defects **early**
* Minimize **reprocessing and waste**
* Save resources and energy

It directly contributes to **Green Computing** practices and industrial efficiency.

---

## Conclusion

This dataset enables the construction of a **realistic, efficient, and sustainable** automated inspection system for the automotive industry.

It serves as the foundation for AI models like **PatchCore**, capable of **learning normality** and detecting anomalies without the need to classify specific defects.
