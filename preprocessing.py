import pandas as pd
import re


def load_data(path):
    # Load the dataset from the specified path
    try: 
        data = pd.read_csv(path)
        print(f"Data loaded successfully from {path}.")
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return None
    return data


def eda(df):
    # Perform exploratory data analysis
    print(df.head())  # Display the first few rows of the dataset
    print(df.isnull().sum())  # Check for missing values


def preprocess_data(df):
    allowed_chars_regex = r"[^a-z0-9\s\n.,'’?!;:()-]"
    # Preprocess the dataset
    df_clean = df.drop(columns=['Unnamed: 0', 'Title', 'Poet', 'Tags']) # Remove unnecessary columns
    cleaned_poems = []
    i = 0
    for poem in df_clean['Poem']:
        poem = poem.replace('\r\n', '\n').replace('\r','\n') # Normalize line endings
        poem = poem.lower()
        poem = re.sub(r'<.*?>', '', poem) # Remove HTML tags
        poem = re.sub(allowed_chars_regex, '', poem) # Remove unwanted characters
        lines = poem.splitlines()  # Split the poem into lines
        lines = [line.strip() for line in lines if line.strip()]  # Remove trailing spaces from each line
        clean_poem = '\n'.join(lines) # Join the verses back into a single string
        if clean_poem:  # Only add non-empty poems
            cleaned_poems.append(clean_poem)
    return "\n\n".join(cleaned_poems)  # Join the poems into a single string


def save_preprocessed_data(string, output_path):
    # Save the preprocessed data to a text file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(string)
    print(f"Preprocessed data saved to {output_path}.")


def main():
    # Read dataset
    path = "PoetryFoundationData.csv"
    df = load_data(path)
    if df is None:
        return
    
    # Perform exploratory data analysis
    eda(df)

    # Preprocess dataset
    string_data = preprocess_data(df)
    
    # save the preprocessed data to a file txt
    output_path = "PoetryFoundationData.txt"
    save_preprocessed_data(string_data, output_path)

if __name__ == "__main__":
    main()