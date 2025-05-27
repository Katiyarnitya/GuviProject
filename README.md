🎵 Mood vs Music 🎧
A Data Analytics project that uses machine learning to classify songs into moods — Happy, Sad, Calm, or Energetic — based on audio features like valence, energy, and tempo.

💡 Project Overview
Mood vs Music is a data analytics and machine learning project designed to predict the mood of a song based on its audio features.

Using a Random Forest Classifier, the project identifies patterns in song data and classifies each track into one of the following categories:

😊 Happy

😢 Sad

😌 Calm

⚡ Energetic

This can be useful for music recommendation engines, playlist generation, and understanding listener sentiment.

🛠️ Technologies Used
Python 🐍

Pandas

NumPy

Matplotlib & Seaborn

Scikit-learn (Machine Learning)

VS Code

📂 Dataset
The dataset contains audio features of songs such as:

Valence: Musical positiveness (0.0 - sad, 1.0 - happy)

Energy: Intensity and activity of the track

Tempo: Beats per minute

Spotify datasets from platforms Kaggle.

⚙️ How It Works
Data Preprocessing

Load the dataset (CSV)

Add a mood label using rule-based classification

Visualization

Display mood distribution using bar charts

Model Building

Train a Random Forest Classifier on valence, energy, and tempo

Evaluation

Display classification report and confusion matrix

Mood Prediction

Predict mood of new songs based on input values

📊 Results
The model achieved good accuracy in classifying moods. Below is a sample confusion matrix and prediction result for a new input.
