# BERT-Based Sentiment Polarity Classification

This project is a sentiment polarity classification task using the BERT (Bidirectional Encoder Representations from Transformers) model. The goal is to classify movie review snippets as either positive or negative, using a pre-trained BERT model fine-tuned for this specific task. The dataset used consists of 5,331 positive and 5,331 negative movie review snippets.

The final model achieves approximately 85% accuracy, with balanced precision and recall metrics.


## Requirements

To run the code, you'll need the following software and packages:

1. **Python 3.x** (I used Python 3.11)
2. **PyTorch** for BERT-based model
3. **HuggingFace Transformers library** for model and tokenizer
4. **Datasets** library for dataset handling
5. **scikit-learn** for calculating metrics
6. **numpy** for saving and loading predictions and labels

### Installation
You can install the required packages by running the following command:

Alternatively, here’s the list of necessary packages:

torch
transformers
datasets
scikit-learn
numpy

```bash
pip install -r requirements.txt



#### 4. **Model and Code Structure**:

main.py: Main script to fine-tune the BERT model, evaluate it, and save predictions and labels.
utils.py: Contains helper functions for loading the dataset.
confusion.py: Script to compute the confusion matrix and various metrics based on saved predictions and labels.
requirements.txt: List of required packages for the project.
rt-polaritydata/: Folder containing the positive and negative dataset files (rt-polarity.pos and rt-polarity.neg).
results/: Folder containing the model checkpoints



#### 5. **How to Run**:

git clone https://github.com/ErenFreedom/BERT_NLP_Assignment1
cd BERT_NLP_Assignment1


Install Dependencies:
pip install -r requirements.txt


You can fine-tune the model, evaluate it, and save predictions by running: python3 main.py
This will load the dataset, fine-tune the BERT model, and save the predictions in predictions.npy and the test labels in test_labels.npy.


To compute the confusion matrix and metrics like precision, recall, and F1 score: python3 confusion.py

**Important:** There is no need to run `main.py` as it would take approximately 1.5 hours to complete training and evaluation.

Instead, the evaluation metrics (accuracy, precision, recall, F1-score) can be computed by running the provided `confusion.py` script. The necessary `.npy` files (`predictions.npy` and `test_labels.npy`) are already available in the repository.

**To get the evaluation results:**
1. Ensure `predictions.npy` and `test_labels.npy` are in the correct directory.
2. Run `confusion.py`:
   ```bash
   python3 confusion.py




#### 6. **Results**:
Accuracy: 84.96%
Precision: 84.38%
Recall: 85.80%
F1-Score: 85.08%


Dataset
The dataset consists of 5,331 positive and 5,331 negative movie reviews from Rotten Tomatoes,provided as part of the rt-polaritydata set. Each sample is a single movie review snippet, downcased.
https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz.

