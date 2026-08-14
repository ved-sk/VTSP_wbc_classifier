# VTSP_wbc_classifier

# White Blood Cell Classifier — Initial Run

## Project Overview

This project investigates the use of transfer learning for automated classification of white blood cells (WBCs) from microscopic images.

The initial model uses **MobileNetV2** as a pretrained convolutional neural network (CNN) backbone. The task is to classify images into four WBC categories:

* Eosinophil
* Lymphocyte
* Monocyte
* Neutrophil

The initial run establishes a baseline against which later CNN architectures can be compared.

---

## How to Run the Notebook

Run the notebook from **top to bottom** in Google Colab.

### 1. Set up the environment

Enable a GPU in:

`Runtime → Change runtime type → T4 GPU (or another available GPU)`

Install/import the required packages.

### 2. Mount/access the dataset

Mount Google Drive if the dataset is stored there and verify that the dataset path is correct.

### 3. Load the images

Load the WBC image dataset and resize the images to:

`224 × 224 × 3`

The four class labels are encoded consistently across the dataset.

### 4. Create the data splits

Create separate training, validation, and test datasets.

The same split should be preserved when comparing architectures so that differences in performance are attributable primarily to the model rather than to different images being used for evaluation.

### 5. Apply data augmentation

The initial pipeline uses:

* Random horizontal flipping
* Random rotation
* Random zoom
* Random contrast
* Random translation

Augmentation is applied to training images rather than evaluation images.

### 6. Build the MobileNetV2 model

The model uses pretrained MobileNetV2 as the feature-extraction backbone, followed by:

* Global Average Pooling
* Dropout
* Dense output layer with four classes

The model is trained using sparse categorical cross-entropy with logits.

### 7. Train the model

Training uses the training dataset with the validation dataset monitored during training.

Early stopping is used to restore the best-performing weights.

### 8. Evaluate the final model

After training, evaluate the model on the held-out test dataset.

Generate:

* Test accuracy
* Confusion matrix
* Classification metrics, if available

---

## Initial Results

The initial MobileNetV2 model achieved approximately **65% held-out test accuracy**.

Training accuracy reached approximately **94–95%**, indicating that the model was able to fit the training data substantially better than it generalized to unseen images.

This difference between training and test performance indicated significant generalization limitations/overfitting.

### Main Confusions

The primary difficulty observed in the initial model was distinguishing **neutrophils from eosinophils**.

The confusion matrix indicated that the model frequently confused these two granulocyte classes.

This suggested that overall accuracy alone was insufficient to characterize the model's performance: errors were concentrated disproportionately among particular WBC categories.

---

## Initial Interpretation

The MobileNetV2 results established a functional baseline but left substantial room for improvement.

The main question became whether a different CNN architecture could produce better generalization and more reliable classification of the visually challenging WBC classes.

---

## Week 4 Improvement Hypothesis

### Hypothesis

A more capable CNN architecture may improve WBC classification performance by producing a more effective learned representation of the input images.

**EfficientNetB0** was selected as the next architecture to investigate because it provides a different CNN architecture while remaining practical for transfer learning in the available computational environment.

The comparison will use the same general classification task and data split as the MobileNetV2 baseline.

### Planned Evaluation

The EfficientNetB0 experiment will be evaluated using:

* Training accuracy
* Validation accuracy
* Held-out test accuracy
* Confusion matrix
* Class-specific performance metrics

The primary question is whether EfficientNetB0 improves overall generalization and whether the improvement is distributed equally across WBC classes.
