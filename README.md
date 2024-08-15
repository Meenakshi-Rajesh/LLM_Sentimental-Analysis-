# LLM_Sentimental-Analysis-

# Overview
This repository contains all the code, datasets, and documentation required to reproduce the fine-tuning process of a BERT model on sentiment analysis tasks. The model has been trained and evaluated using the "Amazon Reviews Multi - German" dataset, which provides a diverse set of product reviews labeled by sentiment. The project is structured to facilitate easy experimentation and further improvements.

# Key Features
-**BERT Fine-Tuning**: Leverages the bert-base-german-cased variant of BERT for effective sentiment analysis.

-**Multilingual Capabilities**: Although focused on German, the methodology can be extended to other languages using appropriate pre-trained models.

-**Efficient Training Pipeline**: Optimized for fast and accurate training, utilizing advanced techniques such as the AdamW optimizer and Sparse Categorical Crossentropy.

-**Comprehensive Evaluation**: Includes tools for evaluating model performance through accuracy metrics and real-world sentiment predictions.


# Data
The Amazon Reviews dataset is sourced from the Hugging Face Datasets library, a comprehensive repository of preprocessed and easily accessible datasets. The specific dataset used is the "amazon_reviews_multi" dataset, which contains multilingual product reviews, including German, English, French, and more. Consist of two primary files: train and validation dataset.Review in this dataset is 
-ID- Unique Identifier

-**Text**- The actual content of the review written by the customer. 

-**Rating**- A numerical rating (1 to 5 stars) given by the customer, which reflects their sentiment towards the product.

-**Label**- A derived label from the star rating that classifies the review as positive, neutral, or negative


# Exploratory Data Analysis



# Pre_Processing
Before feeding the data into the BERT model, several preprocessing steps were necessary:

-**Text Cleaning**: Removing any unwanted characters, special symbols, or formatting issues that could hinder the model's understanding.

-**Tokenization**: Utilizing the BERT tokenizer to break down each review into tokens that the BERT model can process. BERT’s tokenizer handles subtleties in language by breaking words down into smaller components when necessary.

-**Label Encoding**: Converting sentiment labels into a format (integer encoding) that the model can use during training.

-**Dropping Columns**: Dreopping irrelavant columns like ID because these are unique values.

-**Imputation**: To fill Null Values.


# Training and Fine-Tuning
Fine-tuned the pre-trained BERT model (bert-base-german-cased) on the German Amazon Reviews dataset to adapt it for sentiment analysis. Fine-tuning involved adjusting the model on our specific dataset after it had been pre-trained on a large German text corpus. Used the corresponding BERT tokenizer to preprocess the text, converting it into a format suitable for the model. The training process was managed using the Adam optimizer and Sparse Categorical Crossentropy loss function. The model was trained for three epochs, which was sufficient to achieve strong performance without overfitting, as confirmed by the validation accuracy.


# Model Evaluation
After fine-tuning, the model's performance was evaluated on a separate validation dataset. Focused on accuracy as the primary metric to assess how well the model generalizes to unseen data. The final validation accuracy was strong, indicating the model's effectiveness in correctly predicting sentiments in the validation set. Also analyzed the training and validation loss and accuracy curves to monitor the model’s learning progression, ensuring that it was neither overfitting nor underfitting.


# Prediction
The fine-tuned model was tested on new, unseen text inputs to assess its practical performance. A predict_sentiment function was developed, which tokenizes the input text, processes it through the model, and classifies the sentiment based on the highest probability. The model accurately predicted the sentiments of various sample texts, such as positive, neutral, and negative reviews, demonstrating its ability to handle real-world sentiment analysis tasks in German. This confirms the model’s reliability and usefulness in practical applications.

