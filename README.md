Sentiment Analysis on IMDB Reviews 🎬📊
📌 Project Overview

This project performs sentiment analysis on IMDB movie reviews, classifying them as Positive or Negative.
We compare classical machine learning models (TF-IDF + classifiers) with deep learning architectures (LSTM, BiLSTM, CNN, GRU), and include an interactive demo using the trained LSTM model.

🛠 Approach
1. Data Preprocessing

Removed HTML tags, URLs, emails, and special characters.

Expanded contractions (e.g., don’t → do not).

Lemmatization using spaCy.

Removed stopwords except negations (not, never).

2. Feature Engineering

Classical ML → TF-IDF vectorization with unigrams & bigrams.

Deep Learning → Tokenization, padding sequences, embedding layer.

3. Model Training

Classical → Logistic Regression, Linear SVM, Random Forest.

Deep Learning → LSTM, BiLSTM, CNN, GRU (with early stopping).

4. Evaluation

Metrics: Accuracy, Precision, Recall, F1-score.

Visualizations: Confusion matrices, training/validation plots.

📦 Tools & Libraries

Python: Pandas, NumPy, Matplotlib, Seaborn

NLP: spaCy, NLTK, BeautifulSoup, WordCloud

ML/DL: scikit-learn, TensorFlow/Keras

Vectorization: TF-IDF, Keras Tokenizer


📊 Results Summary
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	0.894	0.894	0.896	0.895
Linear SVM	0.886	0.889	0.882	0.886
Random Forest	0.861	0.867	0.854	0.860
LSTM	0.875	0.873	0.879	0.876
BiLSTM	0.879	0.878	0.881	0.880
CNN	0.872	0.870	0.875	0.872
GRU	0.878	0.877	0.880	0.879

Best Classical Model → Logistic Regression (F1 ≈ 0.895)
Best Deep Learning Model → BiLSTM (F1 ≈ 0.880)

🖥 LSTM Demo Script

A separate interactive LSTM prediction script allows users to enter a movie review and instantly get the predicted sentiment with confidence score.

How It Works

Loads the trained LSTM model and tokenizer from disk.

Preprocesses the input text (tokenization + padding).

Predicts sentiment probability.

Displays Positive 😀 or Negative 😠 along with the confidence.

Example Run

=== Sentiment Analysis Demo (LSTM) ===

Enter a review (or 'quit' to exit): This film is absolutely awful, but nevertheless
Predicted Sentiment: Negative 😠  |  Confidence: 0.2580

Enter a review (or 'quit' to exit): I got to see this film at a preview and was da
Predicted Sentiment: Positive 😀  |  Confidence: 0.9108

Enter a review (or 'quit' to exit): This adaptation positively butchers a classic
Predicted Sentiment: Positive 😀  |  Confidence: 0.7813

Enter a review (or 'quit' to exit): quit
👋 Goodbye!

Key Takeaways

Classical ML with TF-IDF and Logistic Regression performed slightly better than deep learning in this dataset.

BiLSTM was the best neural model, handling sequential dependencies well.

The demo script provides an engaging way to test the model interactively.
