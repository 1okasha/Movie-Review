# NLP Assignment

Welcome to this project! Below, you'll find information about the purpose and setup of the project.


## Colab Notebook

Access the interactive notebook using the link below:

[Open in Google Colab]([https://colab.research.google.com/drive/10qXJrQKHdP2Ms8iHnFepTtuH8MUGRLDV?usp=sharing](https://colab.research.google.com/drive/1k1FoJUFGDHOrU27o5vM_wBs2J4GETU2M#scrollTo=6YifDy79bliQ))


# **Sentiment Analysis and Topic Modeling on IMDB Reviews**


## **Objective**
1. **Sentiment Analysis**: Classify IMDB reviews as positive or negative using various machine learning and deep learning models.
2. **Topic Modeling**: Identify common themes in reviews, such as acting, screenplay, and music, using Latent Dirichlet Allocation (LDA).



## **Dataset**
- **Name**: IMDB 50k Reviews Dataset
- **Description**: The dataset contains 50,000 movie reviews, with each review labeled as either positive or negative.
- **Location**: Stored locally as `IMDB_Dataset.csv`.
- **Columns**:
  - `review`: Contains the text of the review.
  - `sentiment`: Binary labels indicating positive or negative sentiment.


## **Project Workflow**

### **1. Exploratory Data Analysis (EDA)**
- Visualized key characteristics of the dataset, including review lengths and word distributions.
- Identified trends and patterns in sentiment distribution.

### **2. Text Preprocessing**
- Tokenization, stopword removal, punctuation removal.
- Applied stemming/lemmatization to normalize words.
- Vectorized text data using techniques like TF-IDF and CountVectorizer.

### **3. Sentiment Classification**
- **Machine Learning Models**:
  - Logistic Regression
  - Naive Bayes
  - Decision Tree
- **Deep Learning Models**:
  - LSTM (Long Short-Term Memory)
  - RNN (Recurrent Neural Network)
  - CNN (Convolutional Neural Network)
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Compared performance across models to identify the best approach for sentiment analysis.

### **4. Topic Modeling**
- **Technique**: Latent Dirichlet Allocation (LDA).
- **Process**:
  - Vectorized text data using CountVectorizer.
  - Applied LDA to uncover 5 distinct topics in the reviews.
  - Visualized topics using bar plots and word clouds.
- **Evaluation**:
  - Used the Silhouette Coherence Score to assess clustering quality.
- **Insights**:
  - Topics identified include acting, screenplay, and music.

---

## **Code Files**
- **Sentiment_Analysis_and_Topic_Modeling.ipynb**:
  - Contains all steps, including:
    - EDA
    - Preprocessing
    - Sentiment classification (ML & DL models)
    - Topic modeling (LDA and evaluation)
- **models.py** (Optional):
  - Contains reusable functions for building ML and DL models.
- **visualizations.py** (Optional):
  - Includes visualization functions (e.g., word clouds, topic distributions).


## **Dependencies**
- Python 3.10+
- NLTK
- Scikit-learn
- TensorFlow/Keras
- Matplotlib
- Seaborn

## **Results**
- **Sentiment Analysis**:
  - Achieved highest accuracy with LSTM: **88.5%**.
  - Logistic Regression performed well among ML models with **85.3%** accuracy.
- **Topic Modeling**:
  - Silhouette Coherence Score: **0.42**
  - Uncovered 5 distinct topics, including themes like **acting** and **screenplay**.
  - Visualized top words for each topic using word clouds and bar plots.





