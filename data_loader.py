import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

def load_and_preprocess_data(data_path, max_length=128):
    """
    Loads the dataset from a CSV file, splits it into train, validation, and test sets, 
    and tokenizes the text data using the BERT tokenizer.

    Args:
        data_path: Path to the CSV file containing the data.
        max_length: Maximum length of the input sequences.

    Returns:
        train_texts: List of training sentences.
        val_texts: List of validation sentences.
        test_texts: List of test sentences.
        train_labels: List of training labels.
        val_labels: List of validation labels.
        test_labels: List of test labels.
        tokenizer: The loaded BERT tokenizer.
    """
    df = pd.read_csv(data_path)

    # Assuming 'review' is the column for reviews and 'sentiment' is the label column
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['review'], df['sentiment'], 
        random_state=2021, 
        test_size=0.3, 
        stratify=df['sentiment']
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, 
        random_state=2021, 
        test_size=0.5, 
        stratify=temp_labels
    )

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    return (
        train_texts, val_texts, test_texts, 
        train_labels, val_labels, test_labels, 
        tokenizer, 
        train_encodings, val_encodings, test_encodings
    )
