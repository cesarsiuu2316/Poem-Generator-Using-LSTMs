import datetime
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model


PATH_DATOS = "datos_limpios/Cleaned_Poems_355927.txt" # Path to the cleaned poems dataset
PATH_POEMAS_GENERADOS = "Results/Poemas_Generados.txt" # Path to save generated poems
PATH_EVALUACIONES = "Results/Evaluations.txt" # Path to save evaluations of generated poems
PATH_MODELO = "Modelos/Poem_Generator_Model_1.keras" # Path to save the trained model

# Parameters for splitting the dataset
TEST_SIZE = 0.2 # Percentage of data to use for testing
RANDOM_STATE = 42 # Random seed for reproducibility

# Global parameters to create the model
VOCABULARY_SIZE = 0 # Will be set after vectorization
N_CELLS = 64 # Number of LSTM cells
VECTOR_MAX_LENGTH = 64 # Maximum length of the vector representation

# Parameters for vectorization
SEQUENCE_LENGTH = 120 # Length of each sequence for training
BATCH_SIZE = 128 # Batch size for training

# Training parameters
EPOCHS = 10 # Number of epochs for training
OPTIMIZER = 'adam' # Optimizer to use for training
LOSS_FUNCTION = 'sparse_categorical_crossentropy' # Loss function for training
METRICS = ['accuracy'] # Metrics to evaluate during training

# Generation parameters
TEMPERATURE = 1.0 # Temperature for sampling during text generation, user controlled


# Function to load the dataset from a text file
def load_data():
    try:
        with open(PATH_DATOS, 'r', encoding='utf-8') as file:
            data = file.read()
        print(f"File {PATH_DATOS} opened successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file {PATH_DATOS} was not found.")
        return None
    

# Function to save the model
def save_model(model):
    os.makedirs("Modelos", exist_ok=True) # Ensure the directory exists
    model.save(PATH_MODELO)
    print(f"Model saved to {PATH_MODELO}.")


# Function to retrieve the model from a file
def retrieve_model(model_path):
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}.")
        return model
    else:
        print(f"Error: The model file {model_path} does not exist.")
        return None


# Function to save results to a text file
def save_results_to_file(string, output_path):
    os.makedirs("Results", exist_ok=True)  # Ensure the directory exists
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write('\n' + string)
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")
        return
    print(f"Results data saved to {output_path}.")


# Function to save generated poems to a file
def save_poems(seed_phrase, poem, generation_time):
    lines = []
    lines.append("=" * 50)
    lines.append(f"Seed phrase:\n{seed_phrase}\n")
    lines.append(f"Generated poem:\n{poem}\n")
    lines.append(f"Generation time: {generation_time:.2f} seconds\n")
    string_poem = "\n".join(lines)
    save_results_to_file(string_poem, PATH_POEMAS_GENERADOS)
    print(f"{string_poem}\nSaved to {PATH_POEMAS_GENERADOS}.")
    return string_poem


# Function to save Model parameters and evaluation results
def save_evaluation_results(model, training_time):
    lines = []
    lines.append("=" * 50)
    lines.append("# Model Training Summary")
    lines.append(f"Date: {datetime.now().isoformat()}")
    lines.append(f"Training Time: {training_time:.2f}s")
    lines.append(f"Dataset path: {PATH_DATOS}")
    lines.append("\n## Parameters")
    lines.append(f"TEST_SIZE: {TEST_SIZE}")
    lines.append(f"RANDOM_STATE: {RANDOM_STATE}")
    lines.append(f"VOCABULARY_SIZE: {VOCABULARY_SIZE}")
    lines.append(f"N_CELLS: {N_CELLS}")
    lines.append(f"VECTOR_MAX_LENGTH: {VECTOR_MAX_LENGTH}")
    lines.append(f"SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    lines.append(f"BATCH_SIZE: {BATCH_SIZE}")
    lines.append(f"EPOCHS: {EPOCHS}")
    lines.append(f"OPTIMIZER: {OPTIMIZER}")
    lines.append(f"LOSS_FUNCTION: {LOSS_FUNCTION}")
    lines.append(f"METRICS: {METRICS}")
    lines.append("\n## Model Architecture")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    lines.extend(model_summary)
    lines.append("\n## Training Results")
    for i, (loss, acc) in enumerate(zip(model.history.history.get("loss", []), model.history.history.get("accuracy", [])), 1):
        lines.append(f"Epoch {i}: loss={loss:.4f}, accuracy={acc:.4f}")
    string = "\n".join(lines)
    save_results_to_file(string, PATH_EVALUACIONES)
    print(f"Evaluation results saved to {PATH_EVALUACIONES}.")


# Function to create dataset and vectorization layer
def create_training_dataset(corpus):
    # Create the TextVectorization layer for the entire text
    vectorize_layer = layers.TextVectorization(
        max_tokens=None, # No limit on vocabulary size
        standardize=None, # Data is already cleaned
        split='character', # Split by character
        output_mode='int' # Output as integers
    )
    # Adapt the vectorization layer to the corpus to build the vocabulary
    vectorize_layer.adapt([corpus])
    # Get the vocabulary size and the character set
    vocab = vectorize_layer.get_vocabulary()
    #print(f"Vocabulary Size: {len(vocab)}")
    #print(f"Vocabulary: {vocab}")
    global VOCABULARY_SIZE
    VOCABULARY_SIZE = len(vocab)

    # Vectorize the entire corpus
    vectorized_text = vectorize_layer([corpus])[0]
    #print(f"Vectorized text length: {len(vectorized_text)}")
    #print(f"First 200 vectorized characters: {vectorized_text[:200]}")

    # Create a tf.data.Dataset from the vectorized text list
    char_dataset = tf.data.Dataset.from_tensor_slices(vectorized_text)
    print(f"Dataset specs: {char_dataset.element_spec}")

    # Use the window method to create sliding windows (subdatasets) of sequences
    sequences_datasets = char_dataset.window(
        SEQUENCE_LENGTH + 1, # +1 to get x and y pairs
        shift=1, # Shift by 1 character to create overlapping sequences
        drop_remainder=True # Drop the last incomplete windows
    )

    # Go through each window and flatten them, than join them into a single dataset
    flat_sequences = sequences_datasets.flat_map(
        lambda window: window.batch(SEQUENCE_LENGTH + 1)) # Batch each window into a sequence of characters

    # 6. Create input (X) and target (y) pairs for each sequence
    def split_input_target(chunk):
        input_text = chunk[:-1] # All characters except the last one
        target_text = chunk[1:] # All characters except the first one
        return input_text, target_text

    # Map the split function to each sequence of the dataset, new dataset will contain pairs of (x, y))
    dataset = flat_sequences.map(split_input_target)

    # Shuffle the dataset by taking a random sample of 10,000 elements, than batch it and start the prefetching, which 
    # allows the model to fetch the next batch while training on the current one, iterating over the dataset
    dataset = dataset.shuffle(10000).batch(
        BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    print(f"Dataset created with sequence_length={SEQUENCE_LENGTH}, batch_size={BATCH_SIZE}")
    return dataset, vectorize_layer


# Function to create the model
def create_model():
    # Use None for the sequence length to allow variable-length inputs
    inputs = layers.Input(shape=(None,), dtype=tf.int64)

    # The Embedding layer needs to have mask_zero to ignore padding
    embedding = tf.keras.layers.Embedding(
        input_dim=VOCABULARY_SIZE,
        output_dim=VECTOR_MAX_LENGTH,
        mask_zero=True
    )(inputs)  # input of embedding = to the output of inputs

    # The LSTM layer with the specified cells and return sequences to learn from past steps
    lstm1 = layers.LSTM(
        N_CELLS,
        return_sequences=True
    )(embedding)  # input of lstm = to the output of embedding

    dropout1 = layers.Dropout(0.2)(lstm1)

    lstm2 = layers.LSTM(
        N_CELLS,
        return_sequences=True
    )(dropout1)

    # Layer 3: Dropout for regularization
    dropout2 = layers.Dropout(0.2)(lstm2)

    # The final Dense layer to predict the next token
    outputs = layers.Dense(
        VOCABULARY_SIZE,
        activation="softmax"
    )(dropout2)  # input of outputs = outputs of lstm

    # Define a model with a starting input and output
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Function to train the model
def train_model(model, dataset):
    print("\n--- Training the model ---")
    model.fit(
        dataset, # The dataset created from the vectorized text already batched with x and y pairs
        epochs=EPOCHS, # Number of epochs to train the model
        verbose=1, # Verbose 1 to see the progress bar
        batch_size=BATCH_SIZE, # Batch size ignored here since we already set it in the dataset
    )
    return model


# Function to get the next character based on the predicted probabilities
def sample(preds):
    preds = np.asarray(preds).astype(np.float64)
    if TEMPERATURE == 1.0:
        return np.argmax(preds) # No temperature scaling, just take the argmax
    preds = np.log(preds + 1e-8) / TEMPERATURE
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    print(f"Preds: {preds}")
    next_char_id = np.random.choice(len(preds), p=preds)
    return next_char_id


# Function to generate a poem based on a seed text
def generate_poem(model, vectorize_layer, seed_text="", num_chars_to_generate=500):
    # Get the vocabulary directly from the layer the model was trained on.
    id_to_char = vectorize_layer.get_vocabulary() 
    # Vectorize the seed text into a 2d tensor of shape (1, sequence_length)
    input_ids = vectorize_layer([seed_text])
    # Array of generated characters
    generated_chars = []
    # Create a tensor to hold the current sequence of input IDs
    current_sequence = tf.identity(input_ids)

    for _ in range(num_chars_to_generate):
        # Predict the next character probabilities
        preds = model.predict(current_sequence, verbose=0)
        # Get the probabilities for only the last step
        # preds is of shape (batch_size, sequence_length, vocabulary_size)
        last_step_probs = preds[:, -1, :]
        # last_step_probs is of shape (batch_size=1, vocabulary_size)
        # 1D array of probabilities for the next character removing the batch dimension
        preds_for_sampling = tf.squeeze(last_step_probs, axis=0).numpy()
        # Sample the next ID. This ID is an index for our id_to_char list.
        next_char_id = sample(preds_for_sampling)
        # Directly get the character from the list.
        generated_chars.append(id_to_char[next_char_id])
        # Prepare the input for the next loop
        next_char_id_tensor = tf.constant([[next_char_id]], dtype=tf.int64)
        # Concatenate the current sequence with the next character ID
        # Limit current sequence to avoid exceeding the sequence length
        current_sequence = current_sequence[-(SEQUENCE_LENGTH-1):]
        # Concatenate the current sequence with the next character ID
        current_sequence = tf.concat(
            [current_sequence, next_char_id_tensor], axis=1)

    # Join and print the result
    print("--- End of Generation ---")
    generated_text = "".join(generated_chars)
    poem = seed_text + generated_text
    return poem


def run_poem_generator(model, vectorize_layer):
    print("\n--- Poem Generator ---")
    while (True):
        # Get user input for seed text, number of characters to generate, and temperature
        seed_text = input("Enter a seed text for the poem: ").strip().lower()
        num_chars_to_generate = 200 # Default number of characters to generate
        try: 
            num_chars_to_generate = int(input("Enter the number of characters to generate: "))
        except ValueError:
            print("Invalid input for number of characters. Using default value of 200.")
            num_chars_to_generate = 200

        global TEMPERATURE
        try:
            TEMPERATURE = float(input("Enter the temperature for sampling (default 1.0): ") or 1.0)
        except ValueError:
            print("Invalid input for temperature. Using default value of 1.0.")
            
        # Start time for poem generation
        start_time = time.time()
        # Generate the poem using the model and vectorization layer
        poem = generate_poem(
            model,
            vectorize_layer,
            seed_text=seed_text,
            num_chars_to_generate=num_chars_to_generate,
        )
        # End time for poem generation
        generation_time = time.time() - start_time
        # Save the generated poem to a file
        output = save_poems(seed_text, poem, generation_time)
        print(output)

        if input("Continue (y/n)? ").strip().lower() != "y":
            break


def main():
    # Read dataset
    poems = load_data()
    if poems is None:
        return
    
    # Load or create the model
    model = None
    try:
        model = retrieve_model(PATH_MODELO)
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Create the training dataset and vectorization layer
    dataset, vectorize_layer = create_training_dataset(poems)
    
    # Get the vocabulary and its size from the vectorization layer
    vocabulary = vectorize_layer.get_vocabulary()
    global VOCABULARY_SIZE
    VOCABULARY_SIZE = len(vocabulary)

    if model is None:
        print("Model could not be loaded. Creating new one.")
        # Create the model and compile
        model = create_model()
        # Compile the model with the specified optimizer, loss function, and metrics
        model.compile(optimizer=OPTIMIZER,loss=LOSS_FUNCTION, metrics=METRICS)
        print(model.summary())
        # Start training time
        start_time = time.time()
        # Train the model
        model = train_model(model, dataset)
        # End training time
        training_time = time.time() - start_time
        # Save the model
        save_model(model, PATH_MODELO)
        # Save evaluation results
        save_evaluation_results(model, model.history, training_time)

    # Run the poem generator
    run_poem_generator(model, vectorize_layer)


if __name__ == "__main__":
    main()