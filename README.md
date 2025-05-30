# SentimentAnalysisAirlines

This repository contains code and documentation for a sentiment analysis project on customer feedback directed to U.S. airlines.  
The objective is to classify tweets into **positive**, **neutral**, or **negative** categories using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 📦 Virtual Environment Setup

To create and activate a virtual environment:

```bash
python -m venv Airlines
Airlines\Scripts\activate # windows
source Airlines/bin/activate # linux/mac
pip install -r requirements.txt
```

 To run a specific model, use the following command:

```bash
python SendAI_<model>.py
```

Replace <model> with the desired model name, such as:
XGBoost
MLP
NaiveBayes
LinearSVM
LogisticRegression

Each script is designed to be self-contained and evaluates the performance of its respective model.

📄 Report
A comprehensive explanation of the methodology, data preprocessing, model selection, and results can be found in the report:
📘 Report_Airlines_NLP.pdf


