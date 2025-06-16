import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split


#np.set_printoptions(threshold=np.inf)


def load_data(file_path):
    try: 
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()  
        print(f"File {file_path} opened successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    

def string_to_list_of_poems(string):
    # Split the string into individual poems based on double newlines
    list_of_poems = string.strip().split('\n\n')
    print(f"Number of poems loaded: {len(list_of_poems)}")
    return list_of_poems


def round_down(x, allowed_values):
    return max([val for val in allowed_values if val <= x], default=min(allowed_values))


def get_perfect_length_for_vectorization(poems):
    poem_lengths = [len(poem) for poem in poems]
    max_length = max(poem_lengths)
    min_length = min(poem_lengths)
    avg_length = sum(poem_lengths) / len(poem_lengths)
    print(f"Max length: {max_length}, Min length: {min_length}, Avg length: {avg_length}")
    p85 = int(np.percentile(poem_lengths, 85))
    p90 = int(np.percentile(poem_lengths, 90))
    print(f"85th percentile length: {p85}, 90th percentile length: {p90}")
    allowed_values = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    p90 = round_down(p90, allowed_values)
    return p90


def embed_poems(list_of_poems, vector_max_length=64, max_tokens=None):
    vectorize_layer = layers.TextVectorization(
        max_tokens=max_tokens,
        standardize = None,
        split='character',
        output_mode='int',
        output_sequence_length=vector_max_length
    )
    vectorize_layer.adapt(list_of_poems)
    vocab = vectorize_layer.get_vocabulary()
    print(f'Vocabulary size: {len(vocab)}\n')
    #print(f'First 100 characters in vocabulary: {vocab[:100]}\n')

    vectors = vectorize_layer(list_of_poems)
    #print(f'First poem as vector: {vectors[:10]}\n')
    return vectors, vectorize_layer


def split_dataset_into_train_and_test(vectors, test_size=0.2, random_state=42):
    X = vectors[:, :-1]  # All but the last character
    y = vectors[:, 1:]   # All but the first character
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f'Training set size: {X_train.shape}, Test set size: {X_test.shape}')
    return X_train, X_test, y_train, y_test


def create_model(sequence_length, vocabulary_size, n_cells=128, vector_max_length=128):
    model = Sequential([
        # Embedding layer to convert integer indices to dense vectors
        # With vocabulary size, output dimension for each character, sequence_length of each list value and masking to handle variable-length sequences
        layers.Embedding(input_dim=vocabulary_size+1, output_dim=vector_max_length, mask_zero=True, input_shape=(sequence_length,)),
        # LSTM layer for sequence processing with return_sequences=True to output a sequence of predictions
        layers.LSTM(n_cells, return_sequences=True),
        # Dense layer to output the next character in the sequence
        layers.Dense(vocabulary_size)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        verbose=1,
        batch_size=batch_size,
    )
    model_softmax = Sequential([
        model,
        layers.Softmax()
    ])
    return model_softmax


def save_model(model, moddel_name):
    model.save(moddel_name + '.keras')


def retrieve_model(model_path):
    if os.path.exists(model_path):
        model = keras.saving.load_model(model_path)
        print(f"Model loaded from {model_path}.")
        return model
    else:
        print(f"Error: The model file {model_path} does not exist.")
        return None
    

def sample(preds, temperature=1):
    preds = np.asarray(preds).astype(np.float64)
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds) 
    preds = exp_preds / np.sum(exp_preds)
    next_char_id = np.random.choice(len(preds), p=preds)
    return next_char_id
    

def generate_poem(model, vocabulary, vectorize_layer, seed_text="", num_chars_to_generate=1000, temperature=1.0):
    # Create a mapping from integer IDs back to characters to decode the model's output
    id_to_char = tf.keras.layers.StringLookup(
        vocabulary=vocabulary, invert=True
    )
    # Convert the seed text into a sequence of integer IDs
    input_ids = vectorize_layer([seed_text])
    # The list to hold the generated characters as integer IDs
    generated_ids = []
    model.reset_states() # Clear any previous state
    for i in range(num_chars_to_generate):
        # Predict the next character probabilities
        preds = model(input_ids)
        next_char_id = sample(preds, temperature=temperature)
        # Append the new character ID to our input for the next loop iteration
        input_ids = tf.concat([input_ids, [next_char_id]], axis=1)
        # Also save it for our final output
        generated_ids.append(next_char_id[0].numpy())

    # Convert the list of generated IDs back to text
    generated_text = tf.strings.reduce_join(id_to_char(generated_ids))
    # Print the final result
    print(seed_text + generated_text.numpy().decode('utf-8'))
    print("--- End of Generation ---")


def main():
    # Retrieve the model
    model = retrieve_model("poem_generator_model.keras")
    if model is None:
        print("Model could not be loaded. Creating new one.")

        # Read dataset
        path = "PoetryFoundationData.txt"
        poems = load_data(path)
        if poems is None:
            print("No data to process.")
            return 
        
        # Convert poems to a list of strings
        list_of_poems = string_to_list_of_poems(poems)
        #print(f'First poem as characters: {list_of_poems[:2]}\n')
        
        # Get the perfect length for vectorization
        max_length = get_perfect_length_for_vectorization(list_of_poems)
        #print(f'Perfect length for vectorization: {max_length}\n')
        
        # Preprocess and embed the poems
        vectors, vectorize_layer = embed_poems(list_of_poems, vector_max_length=max_length)
        vocabulary_size = len(vectorize_layer.get_vocabulary())

        # Devide the dataset into training and validation sets
        vectors = vectors.numpy()  # Convert to numpy array for further processing
        X_train, X_test, y_train, y_test = split_dataset_into_train_and_test(vectors, test_size=0.2, random_state=42)
        sequence_length = X_train.shape[1]  # Get the sequence length from the training data

        # Create the model
        base_model = create_model(sequence_length, vocabulary_size, n_cells=128, vector_max_length=128)
        model = train_model(base_model, X_train, y_train, epochs=20, batch_size=32)
        # Save the model
        save_model(model, "poem_generator_model")

    # Generate a poem
    seed_text = input("Enter a seed text for the poem: ")
    num_chars_to_generate = int(input("Enter the number of characters to generate: "))
    temperature = float(input("Enter the temperature for sampling (default 1.0): ") or 1.0)
    
    generate_poem(
        model,
        vectorize_layer.get_vocabulary(),
        vectorize_layer, seed_text=seed_text,
        num_chars_to_generate=num_chars_to_generate,
        temperature=temperature
    )

if __name__ == "__main__":
    main()