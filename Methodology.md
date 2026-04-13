# ✍️ Section 3: Methodology

## 3. Methodology

### 3.1 Overview

We investigate the effect of training-time exposure to out-of-distribution (OoD) data on the behavior of object detectors. Our methodology consists of three main stages: (i) training a baseline object detector on in-distribution (ID) data, (ii) fine-tuning the model using OoD samples treated as background, and (iii) analyzing the resulting models using both quantitative metrics and explainability techniques. The overall goal is to understand how fine-tuning alters model behavior and contributes to reducing hallucinated detections on OoD inputs.

---

### 3.2 Dataset and Preprocessing

For in-distribution training, we use the PASCAL VOC 2012 dataset, which contains annotated object categories commonly used for benchmarking object detection models. The dataset is converted into YOLO format by transforming XML annotations into normalized bounding box representations. A train–validation split is created using standard data partitioning.

For OoD data, we utilize the Caltech 256 dataset, which contains a diverse set of object categories that do not overlap with the ID classes. A subset of 8,140 images is used for fine-tuning. Importantly, these OoD samples are treated as background during training, meaning no foreground object labels are assigned. This ensures that the model learns to suppress objectness scores for semantically similar but out-of-distribution inputs.

---

### 3.3 Model Training

We employ the YOLOv10s architecture implemented using the Ultralytics YOLO framework. The baseline model is trained on the ID dataset for 50 epochs with an input resolution of 640×640, batch size of 64, and default optimization settings. Training is performed using multiple GPUs to accelerate convergence.

The fine-tuned model is initialized from the trained baseline and further trained for 20 epochs using a combined dataset consisting of ID data and OoD samples. During fine-tuning, the input resolution is set to 512×512 with a batch size of 32. The standard YOLO loss function is used without modification.

---

### 3.4 Fine-tuning Strategy

The key idea of our approach is to expose the model to OoD samples during training and explicitly treat them as background. Unlike traditional OoD detection methods that operate at inference time, this strategy modifies the model’s internal representation during training. By assigning OoD samples to the background class, the model is encouraged to produce low objectness scores for regions that do not correspond to valid ID objects.

This process effectively reshapes the decision boundary of the detector. In standard training, the model learns to distinguish foreground objects from background based solely on ID data, leaving OoD regions underconstrained. Fine-tuning introduces additional constraints by penalizing object-like responses on OoD inputs, resulting in a more conservative objectness function. This reduces the likelihood of hallucinated detections when the model encounters unseen data.

---

### 3.5 Evaluation Protocol

To evaluate model behavior under distribution shift, we perform inference on OoD test images using both baseline and fine-tuned models. Predictions are generated with a confidence threshold of 0.25. We measure hallucinated detections by analyzing the number and quality of predicted bounding boxes on OoD inputs.

To assess spatial alignment and relevance, we compute Intersection-over-Union (IoU) between predicted regions and salient regions identified by explainability methods. IoU is evaluated at multiple thresholds (10%, 20%, and 30%) to capture varying levels of localization accuracy.

---

### 3.6 Explainability and Analysis Methods

To understand the behavioral differences between the baseline and fine-tuned models, we employ multiple explainability techniques.

**Grad-CAM.** We use Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the regions of the input image that contribute most to the model’s predictions. Grad-CAM is applied to the last convolutional layer of the network using a customized wrapper compatible with the YOLO architecture. This allows us to analyze whether the model focuses on semantically meaningful regions or irrelevant background features.

**Occlusion Analysis.** We perform occlusion-based analysis by systematically masking parts of the input image and observing the resulting change in prediction confidence. This helps identify which regions are critical for the model’s decision-making process.

**Insertion and Deletion Metrics.** We quantify explanation quality using insertion and deletion curves. The insertion metric measures how quickly the model’s confidence increases as important regions are progressively revealed, while the deletion metric measures how rapidly confidence decreases as important regions are removed. These metrics provide a quantitative assessment of how well the model’s attention aligns with meaningful features.

**Pointing Game.** We further evaluate localization quality using the pointing game metric, which checks whether the most salient point in the Grad-CAM heatmap falls within the predicted object region.

**Frequency Bias Analysis.** To analyze potential biases, we examine the distribution of predicted classes and identify whether the model exhibits a tendency to favor frequently occurring categories. This helps assess whether fine-tuning reduces reliance on spurious correlations present in the training data.

