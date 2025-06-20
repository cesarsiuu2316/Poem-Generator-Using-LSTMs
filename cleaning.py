import pandas as pd
import re
import os


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


def clean_poems(df, max_lines):
    all_clean_lines = []
    max_line_length = 100
    # Define regex for characters to keep (letters, some punctuation, whitespace, and newlines)
    allowed_chars_regex = r"[^a-z\s\n.,'’?!;:()-]"
    # Remove unnecessary columns
    df_clean = df.drop(columns=['Unnamed: 0', 'Title', 'Poet', 'Tags'])

    for poem in df_clean['Poem']:
        # Clean each poem, lowering case, removing various line endings, trailing chars, and unwanted characters
        poem_lines = []
        text = poem.lower().strip()
        text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\u2028', '\n')
        text = re.sub(allowed_chars_regex, '', text)

        # Split the poem into lines and clean each line
        for line in text.split('\n'):
            # Remove leading/trailing whitespace and multiple spaces/tabs
            cleaned_line = re.sub(r'[ \t]+', ' ', line.strip())
            # Add the cleaned non-empty line if it meets the length requirement
            if cleaned_line and len(cleaned_line) > 1 and len(cleaned_line) <= max_line_length:
                poem_lines.append(cleaned_line)

        # Only add non-empty poems separated by a newline
        if poem_lines:
            all_clean_lines.extend(poem_lines)
            all_clean_lines.append("")

    # Remove the last newline if it exists            
    if all_clean_lines and all_clean_lines[-1] == "\n":
        all_clean_lines = all_clean_lines[:-1]

    # Limit the number of lines if specified
    if max_lines:
        all_clean_lines = all_clean_lines[:max_lines]
    return "\n".join(all_clean_lines)


def save_preprocessed_data(string, output_path):
    # Save the preprocessed data to a text file
    os.makedirs("datos_limpios", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(string)
    print(f"Preprocessed data saved to {output_path}.")


def main():
    # Read and load dataset
    path = "PoetryFoundationData.csv"
    df = load_data(path)
    if df is None:
        return
    # Perform exploratory data analysis
    eda(df)
    # Preprocess dataset
    max_lines_list = [10000, 50000, 100000, None]
    for i, max_lines in enumerate(max_lines_list):
        # Data cleaning
        print(f"\nProcessing with max lines: {max_lines}")
        cleanPoems = clean_poems(df, max_lines)
        num_lines = len(cleanPoems)
        # Save the cleaned poems to a text file
        print(f"Saving clean poems with {num_lines} characters")
        output_path = f"datos_limpios/Cleaned_Poems_{num_lines}.txt"
        save_preprocessed_data(cleanPoems, output_path)


if __name__ == "__main__":
    main()