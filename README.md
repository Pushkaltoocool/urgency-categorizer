# Urgency Classifier Using DistilBERT

This repository contains a complete pipeline for training a text classification model that categorizes incident descriptions by urgency level using HuggingFace's DistilBERT. The model distinguishes between "High Urgency", "Medium Urgency", and "Low Urgency" events based on textual descriptions.

## Overview

The project demonstrates how to:
- Preprocess and tokenize textual data.
- Map text descriptions to urgency categories.
- Perform a stratified train-test split to maintain label distribution.
- Fine-tune a pretrained DistilBERT model for sequence classification.
- Evaluate the model using accuracy as the metric.
- Save the trained model and tokenizer.
- Use the model for interactive inference.

## Dependencies

Ensure you have Python 3.7+ installed along with the following packages:

- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [scikit-learn](https://scikit-learn.org/)
- [torch](https://pytorch.org/)

You can install these packages via pip:
pip install transformers datasets scikit-learn torch

## Data
The dataset is defined within the script and consists of 273 text descriptions. Each description is labeled with one of three urgency categories:

- High Urgency (100 examples)
- Medium Urgency (100 examples)
- Low Urgency (73 examples)
These examples include various incidents such as accidents, natural disasters, infrastructure issues, and less urgent matters like noise complaints or minor maintenance problems.

## Project Structure
Data Preparation:
The script first verifies that the number of descriptions matches the number of labels. Labels are then mapped to numerical values for model training.

- Tokenization:
Uses DistilBertTokenizer to tokenize the descriptions with appropriate truncation and padding.

- Dataset Creation:
This dataset was created manually.

- Model Setup:
Loads a pretrained DistilBertForSequenceClassification model with a classification head adjusted for three classes.

- Training:
Employs HuggingFace's Trainer API with custom training arguments (e.g., batch size, learning rate, logging, and evaluation strategy).

- Evaluation:
Accuracy is computed as the evaluation metric using sklearn.metrics.accuracy_score.

- Model Saving:
After training, the model and tokenizer are saved for later use.

- Inference Functions:
Includes functions to classify new descriptions with an optional confidence threshold, as well as an interactive test loop.

## How to Run

- Train the Model:
Run the script to start the training process. This will tokenize the dataset, fine-tune the DistilBERT model, and save the trained model and tokenizer.

- Interactive Testing:
Once training is complete, the script enters an interactive loop. You can type any description to see its assigned urgency category and the model's confidence score. Type stop to exit.

- Inference Details
classify_issue(description):
Takes a text description, tokenizes it, and outputs the predicted category along with its confidence.

- classify_with_threshold(description, threshold=0.8):
Similar to classify_issue, but if the confidence is below the specified threshold, it flags the prediction as "Uncertain" and suggests manual review.

- Interactive Test Function (test_model()):
Prompts users to input a description repeatedly for testing until the user types 'stop'.

- Customization
Training Parameters:
Adjust hyperparameters (e.g., num_train_epochs, learning_rate, batch_size) in the TrainingArguments to better suit your dataset or prevent overfitting.

- Data:
The dataset is hard-coded within the script. For larger or external datasets, modify the data loading section accordingly.

- Threshold:
Modify the threshold in classify_with_threshold to set a desired confidence level for automatic classification.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
- HuggingFace Transformers
- HuggingFace Datasets
- PyTorch
