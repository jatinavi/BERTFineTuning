# BERTFineTuning
Problem Statement -- Classifying sentences into POSITIVE and NEGATIVE by using fine-tuned BERT model. 

# Sentiment Analysis with BERT

This repository contains code for a sentiment analysis model using the BERT architecture.

**Project Structure:**

* **data_loader.py:** Contains functions for loading and preprocessing the dataset (reading from CSV, splitting into train/val/test, tokenizing with BERT tokenizer).
* **model.py:** Defines the `BERT_architecture` class, which extends the pre-trained BERT model with a classification head.
* **train.py:** Contains functions for training and evaluating the model.
* **evaluate.py:** Contains a function for making predictions on the test data.
* **main.py:** The main script that orchestrates the entire process:
    - Loads data and preprocesses it.
    - Creates DataLoaders.
    - Initializes the model, optimizer, and loss function.
    - Trains the model and saves the best model. 
    - Loads the best model and evaluates it on the test data.

**Dependencies:**

- pandas
- numpy
- torch
- transformers
- scikit-learn

**To run the code:**

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt 
Prepare the data:

Create a CSV file named IMDB Dataset.csv with columns "review" and "sentiment" (where "sentiment" is 0 for negative and 1 for positive).
Alternatively, modify the data_loader.py file to read data from a different source.
Run the main script:

Bash
python main.py
 Note:

This code assumes you have a GPU available. If not, you can adjust the device variable in main.py to use the CPU.
You can adjust hyperparameters like batch size, learning rate, and number of epochs to optimize the model's performance.
This is a basic example, and you can further enhance it by:
Implementing early stopping.
Using techniques like grid search or random search to find optimal hyperparameters.
Adding features like data augmentation or regularization.


**Use Cases:**
1 Customer Feedback Analysis:
Classify customer reviews as positive or negative to gauge product or service performance.
Prioritize addressing negative reviews.
2 Social Media Monitoring:
Analyze tweets, posts, or comments to determine sentiment towards brands, products, or events.
Measure public perception during campaigns or crises.
3 Movie Review Aggregators:
Automatically classify and summarize movie reviews to generate audience sentiment scores.
4 Product Recommendations:
Use sentiment classification to improve recommendation systems by considering review polarity.
5 Political Sentiment Analysis:
Monitor sentiment trends around political campaigns, policies, or candidates.
6 Healthcare Feedback:
Analyze patient feedback for healthcare providers to identify areas of satisfaction or concern.
7 Market Research:
Process survey responses to understand consumer sentiment toward marketing strategies or product launches.

