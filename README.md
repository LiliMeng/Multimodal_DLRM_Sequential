# Multimodal_DLRM_Sequential

## Baselines of MovieLens 25M as sequential prediction
the MovieLens 25M dataset can be used in sequential recommendation problems. Sequential recommendation focuses on predicting the next item (e.g., movie) a user is likely to interact with, based on their previous interactions. This is particularly relevant for tasks like session-based recommendations or modeling user behavior over time.

### How MovieLens 25M Can Be Used in Sequential Recommendation

1. **Sequential Data Preparation**:
   - **Temporal Ordering**: The dataset includes timestamps for each user’s ratings, allowing you to order the interactions sequentially. You can use these timestamps to create sequences of user interactions. For example, you might predict the next movie a user will rate based on the sequence of movies they have rated so far.
   - **Session-based Sequences**: You can split a user’s interactions into sessions if there are clear breaks in time. Each session can then be treated as a sequence.

2. **Feature Engineering**:
   - **Time-based Features**: In sequential recommendation, features like the time gap between interactions can be crucial. You might create features that capture the time since the last interaction or the time of day when interactions occur.
   - **Sequential Embeddings**: MovieLens 25M’s user and movie embeddings can be used to represent sequences. Recurrent neural networks (RNNs), Long Short-Term Memory networks (LSTMs), or Transformer models can process these sequences to predict the next item.

3. **Modeling Approaches**:
   - **RNNs and LSTMs**: These models are well-suited for sequential data and can be used to model the sequence of user interactions.
   - **Transformers**: More recently, Transformers have been successfully applied to sequential recommendation tasks due to their ability to capture long-range dependencies in sequences.
   - **Markov Chains**: You can also use Markov Chains to model the probability of transitioning from one item to the next in a sequence.

4. **Evaluation Metrics**:
   - Common evaluation metrics for sequential recommendation include Hit Rate, Mean Reciprocal Rank (MRR), and Precision@k, which focus on how well the model predicts the next item in the sequence.

### Example Workflow

1. **Data Preparation**:
   - Sort the interactions for each user by timestamp.
   - Create sequences of interactions, with the goal of predicting the next movie a user will rate.

2. **Model Training**:
   - Train a sequential model (e.g., LSTM, Transformer) on the sequences, using previous items in the sequence to predict the next item.

3. **Evaluation**:
   - Evaluate the model using metrics like Hit Rate or MRR to assess how well it predicts the next movie in a sequence.

### Use Case in Research

Sequential recommendation on datasets like MovieLens 25M is useful for applications where the order of user interactions matters, such as in streaming services (predicting the next show to watch) or e-commerce (predicting the next item to buy). Researchers often use MovieLens 25M to benchmark sequential recommendation algorithms, as the large size and rich user interaction data make it an ideal dataset for this purpose.

In summary, while the MovieLens 25M dataset is traditionally used for collaborative filtering, it is well-suited for sequential recommendation tasks, thanks to its detailed interaction data and timestamps.

### LSTM Baseline code
Here’s a step-by-step example of how you could use the MovieLens 25M dataset for a sequential recommendation problem. The goal here is to predict the next movie a user is likely to rate based on their past movie ratings.

### 1. **Load and Prepare the Data**

First, we load the MovieLens 25M dataset and prepare the data by sorting user interactions by timestamp to create sequences.

```python
import pandas as pd
import numpy as np

# Load the MovieLens 25M dataset
ratings = pd.read_csv('ml-25m/ratings.csv')

# Sort ratings by user ID and timestamp to create sequences
ratings = ratings.sort_values(by=['userId', 'timestamp'])

# Create a sequence of movie IDs for each user
user_sequences = ratings.groupby('userId')['movieId'].apply(list).reset_index()
```

### 2. **Create Training and Test Sets**

We split the sequences into training and test sets. Typically, we might use the first part of the sequence for training and the last few interactions for testing.

```python
train_sequences = []
test_sequences = []

# For each user sequence, split into training and testing
for seq in user_sequences['movieId']:
    if len(seq) > 1:  # Only consider sequences with more than 1 interaction
        train_sequences.append(seq[:-1])
        test_sequences.append(seq[-1])
```

### 3. **Modeling with RNN/LSTM**

We use an LSTM model to predict the next movie a user will interact with, given their interaction history.

#### Data Preparation for LSTM

We need to encode the movie IDs and pad the sequences to the same length.

```python
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Encode movie IDs to integers
movie_encoder = LabelEncoder()
all_movie_ids = [movie for seq in train_sequences for movie in seq]
movie_encoder.fit(all_movie_ids)

train_sequences_encoded = [movie_encoder.transform(seq) for seq in train_sequences]
train_sequences_padded = pad_sequences(train_sequences_encoded, padding='pre')

# Prepare input (X) and output (y)
X_train = train_sequences_padded[:, :-1]
y_train = train_sequences_padded[:, 1:]
y_train = np.expand_dims(y_train, -1)  # LSTM expects a 3D input
```

#### Build and Train the LSTM Model

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Define the LSTM model
model = Sequential([
    Embedding(input_dim=len(movie_encoder.classes_), output_dim=50, input_length=X_train.shape[1]),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    Dense(len(movie_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)
```

### 4. **Evaluate the Model**

To evaluate the model, predict the next movie in the sequence for each user in the test set and compute metrics like Hit Rate, MRR, or Precision@k.

```python
def predict_next_movie(model, user_sequence):
    encoded_sequence = movie_encoder.transform(user_sequence)
    padded_sequence = pad_sequences([encoded_sequence], maxlen=X_train.shape[1], padding='pre')
    predictions = model.predict(padded_sequence)
    predicted_movie = np.argmax(predictions, axis=-1)
    return movie_encoder.inverse_transform(predicted_movie)

# Example evaluation: Predict the next movie for the first user
next_movie_predicted = predict_next_movie(model, train_sequences[0])
print(f"Next movie predicted: {next_movie_predicted}")
```

### 5. **Advanced Evaluation with Metrics**

You can implement more sophisticated evaluation methods such as nDCG, Precision@k, or Recall@k, which are common in sequential recommendation systems.

### Summary

This example demonstrates how to turn the MovieLens 25M dataset into a sequential recommendation problem using an LSTM model. The steps include loading and preparing the data, building and training a model, and finally evaluating the model's predictions. This approach allows for the modeling of user behavior over time, making it suitable for applications where the sequence of interactions is important.

## Classification problem
The problem described using the MovieLens 25M dataset for sequential recommendation is a **classification problem**, not a regression problem.

### Why It’s a Classification Problem

1. **Output**: The task involves predicting the next movie that a user will interact with, which means selecting one movie from a set of possible movies. The model's job is to classify the next movie out of all possible movies in the dataset, making it a multi-class classification problem.

2. **Model Output**: The output of the LSTM model (or any sequential model used in this context) is a probability distribution over all possible movies. The movie with the highest probability is chosen as the prediction. This is typical in classification problems.

3. **Loss Function**: In the code provided, the `sparse_categorical_crossentropy` loss function is used. This is a loss function specific to classification tasks, where the goal is to minimize the difference between the predicted class (in this case, the next movie) and the actual class.

### Contrast with Regression

- **Regression**: If the task were to predict a continuous value (like a rating between 0.5 and 5 stars), then it would be a regression problem. For example, using DLRM on MovieLens to predict exact user ratings is a regression task.

In summary, predicting the next movie in a sequence based on previous user interactions is a classification problem because the output is a discrete label (the movie ID) rather than a continuous value.

## Sequential DLRM multimodal
the sequential recommendation problem can still incorporate DLRM (Deep Learning Recommendation Model) and multimodal features such as those extracted using CLIP (Contrastive Language-Image Pre-training). Here's how this can be done:

### Integrating DLRM with Sequential Recommendation and Multimodal Features

1. **Sequential Recommendation with DLRM**:
   - DLRM can be adapted to handle sequential data by incorporating temporal aspects of user interactions. You can do this by feeding in sequences of user interactions into the DLRM model, where each interaction includes both the user’s historical preferences (captured through embeddings) and the multimodal content features of the items (movies) they interacted with.

2. **Using CLIP for Multimodal Features**:
   - **Images**: For each movie, you can use CLIP to generate embeddings from the movie poster.
   - **Text**: CLIP can also be used to generate embeddings from the plot summary or other textual content associated with the movie.
   - These multimodal embeddings are then combined with the embeddings of the user’s historical interactions (like user ID, previous movie IDs) and fed into the DLRM model.

### Step-by-Step Approach

1. **Feature Extraction**:
   - Use CLIP to extract embeddings for both the movie posters and text (like plot summaries).
   - Combine these embeddings with other features such as movie genres, user IDs, and historical interaction data.

2. **Sequence Modeling**:
   - The interaction history of a user (sequence of movies watched or rated) can be modeled using DLRM.
   - Instead of just feeding a single user-item pair, you provide the model with a sequence of interactions, where each interaction includes multimodal features.

3. **Embedding and Interaction**:
   - **Sparse Features**: User ID, Movie ID.
   - **Dense Features**: Temporal features, such as time between interactions.
   - **Multimodal Features**: Embeddings from CLIP for each movie in the sequence.
   - Combine these embeddings using DLRM’s interaction layer, which captures the relationships between the different feature types.

4. **Prediction**:
   - The model predicts the next item (movie) in the sequence that the user is likely to interact with.
   - The prediction could be treated as a classification problem (which movie out of all possible movies) or as a ranking problem, where the model ranks all possible movies and the top-ranked movie is recommended.

### Example Workflow

1. **Prepare Data**: Convert your sequential interaction data into input sequences where each sequence includes user interaction history, multimodal content embeddings, and other relevant features.

2. **Model Design**:
   - The DLRM architecture can be extended to accept sequences of features as input, with additional layers to process the sequence data (e.g., using LSTM or Transformer layers before the interaction layer).
   - Alternatively, you can use the sequence directly in the interaction layer of DLRM if the sequences are short.

3. **Training**:
   - Train the model using a classification loss function if you're predicting the next item in the sequence or using ranking-based loss functions if you aim to rank all potential next items.

4. **Evaluation**:
   - Evaluate the model using metrics suitable for sequential recommendation, such as nDCG, Hit Rate, or MRR, in addition to the usual classification or regression metrics.

### Summary

By combining DLRM with sequence modeling techniques and incorporating multimodal features extracted with models like CLIP, you can build a powerful sequential recommendation system. This approach allows the system to leverage both the temporal patterns in user behavior and the rich multimodal content associated with the items being recommended.



## Code
Implement a sequential recommendation system using DLRM, incorporating multimodal features extracted using CLIP.

### 1. **Install Necessary Libraries**

First, make sure you have all the required libraries installed:

```bash
pip install pandas torch transformers clip-by-openai
```

### 2. **Load and Prepare the Data**

This code snippet loads the MovieLens 25M dataset and extracts multimodal features using CLIP.

```python
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader

# Load the MovieLens 25M dataset
ratings = pd.read_csv('ml-25m/ratings.csv')
movies = pd.read_csv('ml-25m/movies.csv')
links = pd.read_csv('ml-25m/links.csv')

# Sort ratings by user ID and timestamp to create sequences
ratings = ratings.sort_values(by=['userId', 'timestamp'])
user_sequences = ratings.groupby('userId')['movieId'].apply(list).reset_index()

# Load the CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def extract_clip_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    return image_features.squeeze(0).cpu().numpy()

# Example: Extract CLIP features for all movie posters
# (Assume you have movie posters paths linked to movie IDs)
# posters = {movieId: 'path_to_poster_image.jpg'}
# clip_features = {movieId: extract_clip_features(posters[movieId]) for movieId in movies['movieId'].unique()}
```

### 3. **Prepare Sequential Data for DLRM**

Now, encode movie IDs, create sequences, and prepare the dataset for sequential DLRM.

```python
# Encode movie IDs to integers
movie_encoder = LabelEncoder()
all_movie_ids = [movie for seq in user_sequences['movieId'] for movie in seq]
movie_encoder.fit(all_movie_ids)

# Encode sequences
sequences_encoded = [movie_encoder.transform(seq) for seq in user_sequences['movieId']]
sequences_padded = pad_sequences(sequences_encoded, padding='pre')

# Prepare DataLoader
class MovieLensCLIPDataset(Dataset):
    def __init__(self, sequences, clip_features):
        self.sequences = sequences
        self.clip_features = clip_features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_sequence = sequence[:-1]
        target_movie = sequence[-1]

        # CLIP features for the input sequence (last movie in the sequence)
        input_clip_features = self.clip_features.get(target_movie, torch.zeros(512))

        return input_sequence, input_clip_features, target_movie

clip_features = {}  # Dictionary to store CLIP features for each movie
for movie_id in movies['movieId'].unique():
    # Example path handling
    # image_path = f"posters/{movie_id}.jpg"
    # clip_features[movie_id] = extract_clip_features(image_path)
    pass  # Fill in with the actual image path handling code

dataset = MovieLensCLIPDataset(sequences_padded, clip_features)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 4. **Design and Train the DLRM Model with CLIP Features**

This is an example of a DLRM model that incorporates CLIP embeddings as part of the input features.

```python
import torch.nn as nn

class DLRM_CLIP_Sequential(nn.Module):
    def __init__(self, embedding_dim, num_movies, clip_embedding_dim=512, hidden_dim=128):
        super(DLRM_CLIP_Sequential, self).__init__()
        
        # Embedding layers for user and movie sequences
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # MLP for interaction layer
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + clip_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_movies)  # Output layer for classification
        )

    def forward(self, input_sequence, input_clip_features):
        movie_embeds = self.movie_embedding(input_sequence)
        movie_embeds_mean = movie_embeds.mean(dim=1)  # Average movie embeddings in sequence

        combined_features = torch.cat([movie_embeds_mean, input_clip_features], dim=1)
        output = self.mlp(combined_features)
        return output

# Instantiate and train the model
num_movies = len(movie_encoder.classes_)
embedding_dim = 128
model = DLRM_CLIP_Sequential(embedding_dim, num_movies).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for input_sequence, input_clip_features, target_movie in dataloader:
        input_sequence, input_clip_features, target_movie = input_sequence.to(device), input_clip_features.to(device), target_movie.to(device)

        optimizer.zero_grad()
        output = model(input_sequence, input_clip_features)
        loss = criterion(output, target_movie)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
```

### 5. **Evaluation**

To evaluate the model, you would use the nDCG, Hit Rate, or similar metrics to measure how well it predicts the next movie in the sequence.

```python
def evaluate_model(model, dataloader):
    model.eval()
    total_hits = 0
    total_items = 0

    with torch.no_grad():
        for input_sequence, input_clip_features, target_movie in dataloader:
            input_sequence, input_clip_features, target_movie = input_sequence.to(device), input_clip_features.to(device), target_movie.to(device)

            output = model(input_sequence, input_clip_features)
            predicted_movie = output.argmax(dim=1)
            
            total_hits += (predicted_movie == target_movie).sum().item()
            total_items += target_movie.size(0)

    hit_rate = total_hits / total_items
    print(f"Hit Rate: {hit_rate}")

# Call evaluation function
evaluate_model(model, dataloader)
```

### Summary

This example shows how to implement a sequential recommendation system using DLRM, where the system leverages multimodal features extracted by CLIP. The model combines user interaction sequences with movie embeddings and multimodal content features to predict the next movie a user might engage with. This approach can be extended with additional features or more sophisticated sequence modeling techniques like LSTMs or Transformers, depending on the specific requirements.
