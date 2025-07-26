# Sentiment Analysis with NLP - Task 2
## CODTECH Internship Project

This project demonstrates sentiment analysis on customer reviews using TF-IDF vectorization and logistic regression classification.

## 🎯 Objectives

- Perform sentiment analysis on customer review data
- Implement TF-IDF vectorization for text feature extraction
- Train a logistic regression model for sentiment classification
- Evaluate model performance using various metrics
- Create an end-to-end sentiment analysis pipeline

## 📊 Features

- **Text Preprocessing**: Comprehensive text cleaning and normalization
- **TF-IDF Vectorization**: Convert text to numerical features
- **Machine Learning**: Logistic regression for classification
- **Model Evaluation**: Detailed performance analysis with visualizations
- **Sentiment Prediction**: Test model on new customer reviews

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and metrics
- **NLTK**: Natural language processing
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 📁 Project Structure

```
sentiment_analysis/
├── sentiment_analysis.ipynb       # Main notebook with complete analysis
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .vscode/
│   └── tasks.json                 # VS Code tasks configuration
└── .github/
    └── copilot-instructions.md    # Copilot customization
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   cd sentiment_analysis
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data (run this once in Python):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Running the Project

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Or use the VS Code task: `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Jupyter Notebook"

2. Open `sentiment_analysis.ipynb` in the Jupyter interface

3. Run all cells sequentially to see the complete analysis

**Note**: The notebook creates sample customer review data internally, so no external dataset is required.

## 📈 Results

The project demonstrates:
- Effective text preprocessing techniques
- TF-IDF feature extraction from customer reviews
- Logistic regression model training and evaluation
- Comprehensive performance analysis with visualizations
- Practical sentiment prediction on new reviews

## 🔄 Workflow

1. **Data Creation**: Generate sample customer review dataset (25 reviews)
2. **Data Exploration**: Analyze sentiment distribution and review characteristics
3. **Text Preprocessing**: Clean and normalize text data using NLTK
4. **Vectorization**: Apply TF-IDF transformation to convert text to numerical features
5. **Model Training**: Train logistic regression classifier with train/test split
6. **Evaluation**: Assess model performance with accuracy, classification report, and confusion matrix
7. **Visualization**: Create performance visualizations and feature importance plots
8. **Prediction**: Test model on new, unseen reviews

## 📊 Model Performance

The model demonstrates:
- **Accuracy**: Classification accuracy on test data
- **Classification Report**: Precision, recall, and F1-score for each sentiment class
- **Confusion Matrix**: Visual representation of prediction accuracy
- **ROC Curve**: Receiver Operating Characteristic curve analysis
- **Feature Importance**: Most influential words for sentiment classification
- **Sample Predictions**: Confidence scores for new review predictions

## 🔧 Customization

You can modify the project by:
- **Dataset**: Replace sample data with your own customer review dataset
- **Preprocessing**: Adjust text cleaning parameters and add custom preprocessing steps
- **TF-IDF Parameters**: Modify max_features, ngram_range, min_df, max_df settings
- **Algorithms**: Experiment with different classification algorithms (SVM, Random Forest, etc.)
- **Evaluation**: Add additional metrics like precision-recall curves
- **Visualization**: Enhance plots with more detailed analysis and insights

## 📚 Learning Outcomes

This project demonstrates:
- **NLP Fundamentals**: Text preprocessing, tokenization, and feature extraction
- **Machine Learning**: Classification algorithms and model evaluation
- **TF-IDF Vectorization**: Converting text data to numerical features
- **Scikit-learn**: Using ML library for data science workflows
- **Data Visualization**: Creating informative plots for model analysis
- **Jupyter Notebooks**: Interactive development and data exploration
- **Reproducible Research**: Setting random seeds and following best practices

## 🤝 Contributing

Feel free to fork this project and submit improvements:
- Enhanced text preprocessing
- Additional machine learning models
- Better visualization techniques
- Performance optimizations

## 📄 License

This project is created for educational purposes as part of the CODTECH internship program.

## 👨‍💻 Author

Created as part of CODTECH Internship Task 2 - Sentiment Analysis with NLP

---

**Note**: This is a demonstration project using sample data created within the notebook. The dataset contains 25 customer reviews with positive, negative, and neutral sentiments. For production use, consider using larger, real-world datasets and more sophisticated models.
