# Topic Modelling with LDA and BERT

## 📚 Project Description
This project implements topic modelling using two advanced techniques:
- **Latent Dirichlet Allocation (LDA):** to uncover hidden topics in text data.
- **BERT embeddings:** for semantic clustering and dimensionality reduction using PCA.

The project processes text reviews, assigns topics to each review, and visualizes the results. Evaluation is performed using Perplexity and Coherence Scores.

---

## 🚀 Features
- **LDA Topic Modelling:** Extracts meaningful topics from text data.
- **BERT-based Clustering:** Generates embeddings for semantic clustering.
- **Custom Stopword Removal:** Removes domain-specific and common stopwords.
- **Dimensionality Reduction:** Uses PCA to reduce embeddings for visualization.
- **Model Evaluation:** Calculates Perplexity and Coherence Scores.
- **Data Labeling:** Labels training and testing data based on identified topics.

---

## 🔧 Installation
Ensure you have Python installed, then install the required libraries:

```bash
# Install core libraries
pip install pandas scikit-learn matplotlib gensim transformers torch

# Install optional classifiers
pip install catboost lightgbm xgboost
```

If using GPU for BERT processing, ensure PyTorch is set up correctly:

```bash
# For CUDA-enabled GPUs
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📈 Usage
### 1. Label Training Data
Run the following command to label training data using LDA:

```python
from topic_modelling import assign_labels

df = assign_labels(input_file='csv_data.csv', output_file='training_data.csv', useful_topics=[1, 3], num_clusters=4)
```

### 2. Label Testing Data

```python
testing_df = assign_labels(input_file='unseen_data.csv', output_file='testing_data.csv', useful_topics=[2])
```

### 3. BERT-Based Topic Modelling

```python
# Load BERT embeddings, reduce dimensionality, and cluster
embeddings = get_bert_embeddings(reviews)

# Perform PCA and KMeans clustering
reduced_embeddings = pca.fit_transform(embeddings)
df['topic'] = kmeans.fit_predict(embeddings)

# Visualize results
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['topic'], cmap='viridis', alpha=0.6)
plt.show()
```

---

## 📊 Evaluation
The project calculates the following scores to evaluate topic models:

- **Perplexity Score:** Measures how well the model predicts a sample.
- **Coherence Score:** Evaluates topic coherence based on word similarity.

```python
print(f"Model Perplexity: {perplexity}")
print(f"Model Coherence Score: {coherence_score}")
```

---

## 🤝 Contributing
Contributions are welcome! To get started:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## 🌟 Acknowledgments
Special thanks to open-source contributors and libraries used in this project: Scikit-learn, Gensim, Transformers, and PyTorch.

