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

🛠️ Technologies Used :

Python 🐍

Pandas

NumPy

Matplotlib & Seaborn

Scikit-learn (Machine Learning)

VS Code

📂 Dataset :

The dataset contains audio features of songs such as:

Valence: Musical positiveness (0.0 - sad, 1.0 - happy)
Energy: Intensity and activity of the track
Tempo: Beats per minute
Spotify datasets from platforms Kaggle.

⚙️ How It Works :

1. Data Preprocessing -> Load the dataset (CSV)
                      -> Add a mood label using rule-based classification
2. Visualization -> Display mood distribution using bar charts
3. Model Building -> Train a Random Forest Classifier on valence, energy, and tempo
4. Evaluation -> Display classification report and confusion matrix
5. Mood Prediction -> Predict mood of new songs based on input values

🔹 Description of the Dataset used for Visualization ->
The dataset contains information about various songs, including attributes like valence (positivity), energy, tempo, danceability, and mood labels (Happy, Sad, Calm, Energetic). These musical characteristics help analyze how mood correlates with technical features of songs.

🔹 Summary of Visualizations Used
Mood Distribution (Bar Chart): Count of songs per mood.

Valence vs Energy (Scatter Plot): Shows the relationship between emotion and energy levels.

Average Valence & Energy per Mood (Clustered Bar Chart): Highlights mood-wise musical intensity.

Danceability vs Energy (Scatter/Bubble Chart): Depicts how danceability aligns with energy, scaled by valence.

Tempo Across Moods (Line Chart): Represents variation in tempo among moods.

Energy Over Danceability (Stacked Area Chart): Visualizes how energy is distributed with increasing danceability, grouped by mood.

🔹 Key Insights
Happy songs generally score highest on valence and danceability.

Sad songs show the lowest values for tempo and energy.

Calm moods lie in the mid-range of most attributes.

Energetic songs correlate strongly with high tempo and energy.

Visual patterns reveal how mood labels can be predicted using song features.

🔹 How to View/Interact with the Report
Use the Mood Legend to distinguish between moods across all visuals.
Hover over data points to view exact song names, values, and moods.
Navigate the report using Power BI’s tab or scroll feature if multiple pages exist.
Charts are interactive — clicking on a section in one chart filters relevant values in others (if cross-filtering is enabled

📊 Results
The model achieved good accuracy in classifying moods. Below is a sample confusion matrix and prediction result for a new input.
