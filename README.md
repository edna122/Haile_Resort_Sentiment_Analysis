Hotel Review Analysis for Haile Hotels and Resorts

This project analyzes customer reviews from online platforms to enhance service quality and guest experience at Haile Hotels and Resorts in Ethiopia. Using machine learning techniques, the project identifies key service aspects mentioned in reviews and assesses overall customer sentiment.

Table of Contents

Abstract

Introduction

Statement of the Problem

Objectives

Related Works

Methods

Experimental Description

Results and Discussion

Conclusion and Recommendations

References

Getting Started

Dependencies


Abstract

This project leverages customer review analysis to improve service quality and guest experience at Haile Hotels and Resorts in Ethiopia. Reviews from Booking.com and Hoteles.com were collected and preprocessed. Machine learning models were employed for sentiment analysis and topic identification. The analysis provides actionable recommendations for Haile Hotels and Resorts.

Introduction

Online reviews are crucial for the hospitality industry. This project analyzes these reviews for Haile Hotels and Resorts to understand customer feedback and identify areas for improvement. Six Haile Hotels and Resorts are included in the analysis.

Statement of the Problem

Effective review analysis is essential for hotels to understand customer preferences and maintain a competitive edge. This project addresses the underutilization of review analysis in the Ethiopian hotel industry.

Objectives

Collect and aggregate review data.

Perform sentiment analysis.

Identify key service aspects.

Develop a visualization dashboard.

Provide actionable recommendations.

Related Works

This project draws upon existing research in hotel review analysis, including studies on sentiment classification, aspect-based sentiment analysis, and the impact of online reviews on hotel performance.

Methods

Data Collection: Manual collection from Booking.com and Hoteles.com.

Data Cleaning: Handling missing values, rating scale conversion, text preprocessing.

Tools: Python (Pandas, Scikit-learn, NLTK, spaCy, Gensim, Matplotlib, Seaborn).

Algorithms: Logistic Regression, Decision Trees, Random Forest, AdaBoost (sentiment); LDA, NMF, K-Means, LSA (topic identification).

Evaluation: Accuracy, precision, recall, F1-score, coherence score, silhouette score.

Experimental Description

Sentiment Analysis: Train/test split, model training, hyperparameter tuning.

Topic Identification: Application of topic modeling models, performance evaluation.

Results and Discussion

Logistic Regression model achieved the highest accuracy for sentiment analysis. NMF provided the best results for topic identification. Key topics and sentiment insights are discussed in the report.

Conclusion and Recommendations

The project provides a framework for review analysis and offers specific recommendations for Haile Hotels and Resorts to improve service quality.

References

(List of references as provided in the project report)

Getting Started

Clone the repository: git clone https://github.com/edna122/Haile_Resort_Sentiment_Analysis

Navigate to the project directory: cd YOUR_REPOSITORY_NAME

Install the dependencies (see Dependencies).

Dependencies

pandas
scikit-learn
nltk
spacy
gensim
matplotlib
seaborn
# Add any other dependencies

You can install these using pip:

pip install -r requirements.txt # Create a requirements.txt file with the above dependencies


