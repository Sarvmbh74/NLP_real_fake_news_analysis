Here's your complete and updated `README.md` content with the **Streamlit link** and **accuracy scores** in percentage format:

---

# 📰 Real vs Fake News Analysis using NLP

This project applies Natural Language Processing (NLP) techniques to classify news articles as **real (0)** or **fake (1)**. It involves preprocessing textual data, exploratory data analysis (EDA), model building using multiple machine learning algorithms, model evaluation, and deployment using **Streamlit**.

---

## 📌 Objective

The goal of this project is to build a machine learning model that can accurately detect fake news from real news by analyzing the content of the articles using NLP.

---

## 📂 Project Structure

```
real_fake_news_analysis/
│
├── data/                     # Dataset used for training and testing
├── notebooks/                # Jupyter notebooks for EDA and model building
├── model/                    # Trained model files
├── app/                      # Streamlit app files for deployment
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License file
└── README.md                 # Project description
```

---

## 🧠 Technologies & Tools Used

- **Python**
- **NLP (Natural Language Processing)**
  - Stopwords removal
  - Tokenization
  - Lemmatization/Stemming
- **ML Algorithms**
  - Logistic Regression
  - Naive Bayes
  - Decision Tree Classifier
  - XGBoost (selected for deployment)
- **Libraries**
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `sklearn`, `xgboost`, `nltk`, `re`
  - `streamlit` (for deployment)
- **Model Evaluation**
  - Accuracy, Confusion Matrix, Precision, Recall, F1 Score
- **Deployment**
  - Streamlit Web App

---

## 🔍 Exploratory Data Analysis (EDA)

- Distribution of real vs fake news
- Most frequent words in each category
- Word cloud visualizations
- Text length, punctuation, and stopword usage

---

## 🛠️ Model Building & Evaluation

Four models were trained and evaluated:

| Algorithm             | Accuracy (%) |
|-----------------------|--------------|
| Logistic Regression   | 98.99%       |
| Naive Bayes           | 93.85%       |
| Decision Tree         | 99.65%       |
| **XGBoost (Best)**    | **99.78%**   |

> XGBoost achieved the highest accuracy and was chosen for deployment.

---

## 🚀 Deployment

The final model is deployed using **Streamlit** as a web application, where users can paste a news article and get an instant prediction of whether it's real or fake.

🔗 [Streamlit App Link](https://nlprealfakenewsanalysis-fveqsl8pkrcpc8kr98hfux.streamlit.app/)

---

## 📊 Results

- Final Accuracy (XGBoost): **99.78%**
- Highlights:
  - Efficient preprocessing pipeline
  - Feature extraction using bag-of-words / TF-IDF
  - User-friendly web interface

---

## 👨‍💻 How to Run Locally

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/real_fake_news_analysis.git
   cd real_fake_news_analysis
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app  
   ```bash
   streamlit run app/app.py
   ```

---

## 👥 Team Members

- Sarvmbh Sawant (P422)
- Miss Arati Shinde
- [Add others if any]

---

## 📎 License

This project is licensed under the [MIT License](LICENSE).

© 2025 Sarvmbh Sawant. See the LICENSE file for more details.

---

## 🙌 Acknowledgments

- Kaggle or original dataset source
- NLTK for NLP tools
- Streamlit for deployment support

---

Let me know if you'd like this content saved in a `.md` file again or need help uploading it to GitHub.
