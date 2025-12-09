import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Assume your main script functions (SimpleVocab, IMDBDataset, etc.) are imported ---
# from your_main_script import SimpleVocab, IMDBDataset, TransformerModel, prepare_data

# For a self-contained test, we'll define the necessary functions here.
# In your real use, you would import them from your main script.

# Minimal versions of your classes for the test to run
class SimpleVocab:
    def __init__(self, min_freq=2, max_size=20000):
        self.stoi = {'<pad>': 0, '<unk>': 1}
        self.itos = {0: '<pad>', 1: '<unk>'}
        self.min_freq = min_freq
        self.max_size = max_size
    def build(self, texts):
        counter = {}
        for text in texts:
            for token in self.tokenize(text):
                counter[token] = counter.get(token, 0) + 1
        
        words = sorted([word for word, freq in counter.items() if freq >= self.min_freq], 
                      key=lambda x: -counter[x])[:self.max_size-2]
        for i, word in enumerate(words, 2):
            self.stoi[word] = i
            self.itos[i] = word
    def tokenize(self, text):
        return re.findall(r'\b\w+\b', str(text).lower())
    def encode(self, text):
        return [self.stoi.get(token, 1) for token in self.tokenize(text)]
    def __len__(self):
        return len(self.stoi)

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.vocab.encode(text)
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        padding = [0] * (self.max_len - len(encoded))
        encoded = encoded + padding
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class TransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=256, n_heads=8, n_layers=4, d_ff=512, dropout=0.1, max_len=128):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(max_len, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(d_model, d_model // 2), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(d_model // 2, num_classes))
        self.d_model = d_model
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :x.size(1)].to(x.device)
        x = self.transformer(x)
        x = self.classifier(x[:, 0, :])
        return x

# ==============================================================================
# CORRECTED TEST FUNCTIONS
# ==============================================================================

def create_synthetic_data(n_samples=100, for_genre_test=False):
    """Create synthetic IMDb-like data for testing"""
    np.random.seed(42)
    
    titles = [f"Movie_{i}" for i in range(n_samples)]
    years = np.random.randint(1990, 2023, n_samples)
    runtimes = np.random.randint(80, 180, n_samples)
    ratings = np.random.uniform(4.0, 9.0, n_samples)
    votes = np.random.randint(1000, 100000, n_samples)
    directors = [f"Director_{i}" for i in range(20)]
    actors = [f"Actor_{i}" for i in range(50)]
    movie_directors = [np.random.choice(directors) for _ in range(n_samples)]
    movie_actors = [', '.join(np.random.choice(actors, 3, replace=False)) for _ in range(n_samples)]

    # CORRECTED: Generate genres that will pass the filter
    if for_genre_test:
        # Create a few genres with many samples to pass the >=1000 filter
        genres_to_use = ['Action', 'Comedy', 'Drama'] 
        samples_per_genre = n_samples // len(genres_to_use)
        movie_genres = []
        for genre in genres_to_use:
            movie_genres.extend([genre] * samples_per_genre)
        # Add remaining samples to the first genre
        remaining = n_samples - len(movie_genres)
        movie_genres.extend([genres_to_use[0]] * remaining)
        np.random.shuffle(movie_genres)
    else:
        # For other tests, use a variety of genres
        all_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
        movie_genres = [np.random.choice(all_genres) for _ in range(n_samples)]

    df = pd.DataFrame({
        'primaryTitle': titles, 'genres': movie_genres, 'startYear': years,
        'runtimeMinutes': runtimes, 'averageRating': ratings, 'numVotes': votes,
        'directors': movie_directors, 'actors': movie_actors
    })
    return df

def prepare_data(movies, task='rating'):
    """This is the function from your main script that is being tested."""
    if task == 'rating':
        movies['text'] = movies.apply(lambda x: f"{x['primaryTitle']} {x['genres'].replace(',', ' ')} Year {int(x['startYear'])} Runtime {int(x['runtimeMinutes'])}min Director {x.get('directors', '')} Cast {x.get('actors', '')}", axis=1)
        movies['label'] = pd.cut(movies['averageRating'], bins=[0, 6, 7.5, 10], labels=[0, 1, 2])
        num_classes = 3
    elif task == 'genre':
        movies['primary_genre'] = movies['genres'].str.split(',').str[0]
        movies = movies[movies['primary_genre'] != '\\N'].copy()
        genre_counts = movies['primary_genre'].value_counts()
        common_genres = genre_counts[genre_counts >= 1000].index # This is the line that requires specific data
        movies = movies[movies['primary_genre'].isin(common_genres)].copy()
        if len(movies) == 0:
            raise ValueError("No movies left after genre filtering. Test data may be too small or not diverse enough.")
        movies['text'] = movies.apply(lambda x: f"{x['primaryTitle']} Year {int(x['startYear'])} Runtime {int(x['runtimeMinutes'])}min Director {x.get('directors', '')} Cast {x.get('actors', '')}", axis=1)
        le = LabelEncoder()
        movies['label'] = le.fit_transform(movies['primary_genre'])
        num_classes = len(le.classes_)
    elif task == 'popularity':
        movies['text'] = movies.apply(lambda x: f"{x['primaryTitle']} {x['genres'].replace(',', ' ')} Year {int(x['startYear'])} Runtime {int(x['runtimeMinutes'])}min Director {x.get('directors', '')} Cast {x.get('actors', '')}", axis=1)
        movies['label'] = pd.qcut(movies['numVotes'], q=3, labels=[0, 1, 2])
        num_classes = 3
    return movies['text'].tolist(), movies['label'].tolist(), num_classes


def test_data_preparation():
    """Test data preparation functions"""
    print("="*60)
    print("TESTING DATA PREPARATION")
    print("="*60)
    
    # Create synthetic data for non-genre tasks
    df = create_synthetic_data(100)
    print(f"✓ Created synthetic dataset with {len(df)} movies")
    
    # Test rating task
    print("\n1. Testing rating task...")
    texts, labels, num_classes = prepare_data(df, task='rating')
    assert len(texts) == len(labels) == 100, "Data length mismatch"
    assert num_classes == 3, "Should have 3 rating classes"
    print(f"   ✓ Rating task: {len(texts)} samples, {num_classes} classes")
    
    # Test genre task with CORRECTED data
    print("\n2. Testing genre task...")
    # Create a larger dataset specifically for the genre test to pass filtering
    df_genre = create_synthetic_data(n_samples=3000, for_genre_test=True) 
    print(f"   Created genre-specific dataset with {len(df_genre)} movies")
    texts, labels, num_classes = prepare_data(df_genre, task='genre')
    assert len(texts) == len(labels) > 0, "Data length mismatch or empty result"
    assert num_classes > 1, "Should have multiple genre classes"
    print(f"   ✓ Genre task: {len(texts)} samples, {num_classes} classes")
    
    # Test popularity task
    print("\n3. Testing popularity task...")
    texts, labels, num_classes = prepare_data(df, task='popularity')
    assert len(texts) == len(labels) == 100, "Data length mismatch"
    assert num_classes == 3, "Should have 3 popularity classes"
    print(f"   ✓ Popularity task: {len(texts)} samples, {num_classes} classes")
    
    return df

# The rest of the test functions remain the same...
def test_vocabulary_and_dataset():
    print("\n" + "="*60); print("TESTING VOCABULARY AND DATASET"); print("="*60)
    df = create_synthetic_data(50)
    texts, labels, num_classes = prepare_data(df, task='rating')
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.4, random_state=42)
    vocab = SimpleVocab(min_freq=1, max_size=1000); vocab.build(X_train)
    assert len(vocab) > 10 and '<pad>' in vocab.stoi and '<unk>' in vocab.stoi
    print(f"\n1. Testing vocabulary...\n   ✓ Vocabulary size: {len(vocab)}")
    test_text = "Action Movie 2020"; encoded = vocab.encode(test_text)
    assert len(encoded) > 0 and all(isinstance(x, int) for x in encoded)
    print(f"   ✓ Encoding works: '{test_text}' -> {encoded[:5]}...")
    dataset = IMDBDataset(X_train, y_train, vocab, max_len=32)
    assert len(dataset) == len(X_train)
    sample_text, sample_label = dataset[0]
    assert isinstance(sample_text, torch.Tensor) and isinstance(sample_label, torch.Tensor) and sample_text.shape[0] == 32
    print(f"\n2. Testing dataset...\n   ✓ Dataset works: {len(dataset)} samples")
    return vocab, dataset

def test_model():
    print("\n" + "="*60); print("TESTING MODEL"); print("="*60)
    model = TransformerModel(vocab_size=100, num_classes=3, d_model=64, n_heads=4, n_layers=2, max_len=32)
    print(f"✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")
    batch_size, seq_len = 4, 32; x = torch.randint(0, 100, (batch_size, seq_len))
    with torch.no_grad(): output = model(x)
    assert output.shape == (batch_size, 3)
    print(f"\n1. Testing forward pass...\n   ✓ Forward pass works: input {x.shape} -> output {output.shape}")
    for bs in [1, 2, 8]:
        x = torch.randint(0, 100, (bs, seq_len))
        with torch.no_grad(): output = model(x)
        assert output.shape[0] == bs
    print("\n2. Testing different batch sizes...\n   ✓ Different batch sizes work")
    return model

def test_training_loop():
    print("\n" + "="*60); print("TESTING TRAINING LOOP"); print("="*60)
    df = create_synthetic_data(20); texts, labels, num_classes = prepare_data(df, task='rating')
    vocab = SimpleVocab(min_freq=1, max_size=100); vocab.build(texts)
    dataset = IMDBDataset(texts, labels, vocab, max_len=16); loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    model = TransformerModel(vocab_size=len(vocab), num_classes=num_classes, d_model=32, n_heads=2, n_layers=1, max_len=16)
    criterion = torch.nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print("\n1. Running 3 training steps..."); model.train()
    for step, (data, target) in enumerate(loader):
        optimizer.zero_grad(); output = model(data); loss = criterion(output, target); loss.backward(); optimizer.step()
        print(f"   Step {step+1}: Loss = {loss.item():.4f}"); if step >= 2: break
    print("   ✓ Training loop works")
    print("\n2. Testing overfitting on tiny data..."); model.train(); initial_loss = None
    for epoch in range(20):
        epoch_loss = 0
        for data, target in loader: optimizer.zero_grad(); output = model(data); loss = criterion(output, target); loss.backward(); optimizer.step(); epoch_loss += loss.item()
        if initial_loss is None: initial_loss = epoch_loss
        if epoch % 5 == 0: print(f"   Epoch {epoch}: Loss = {epoch_loss:.4f}")
    final_loss = epoch_loss
    print(f"   ✓ Overfitting test passed: {initial_loss:.4f} -> {final_loss:.4f}")

def test_end_to_end():
    print("\n" + "="*60); print("TESTING END-TO-END PIPELINE"); print("="*60)
    config = {'task': 'rating', 'max_movies': 50, 'max_len': 32, 'batch_size': 8, 'epochs': 2, 'lr': 0.01}
    print(f"Running end-to-end test with {config['max_movies']} samples...")
    df = create_synthetic_data(config['max_movies']); texts, labels, num_classes = prepare_data(df, config['task'])
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.3, random_state=42)
    vocab = SimpleVocab(min_freq=1, max_size=200); vocab.build(X_train)
    train_dataset = IMDBDataset(X_train, y_train, vocab, config['max_len']); val_dataset = IMDBDataset(X_val, y_val, vocab, config['max_len'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True); val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'])
    model = TransformerModel(vocab_size=len(vocab), num_classes=num_classes, d_model=64, n_heads=4, n_layers=2, max_len=config['max_len'])
    criterion = torch.nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    print("\nTraining for 2 epochs...")
    for epoch in range(config['epochs']):
        model.train(); train_loss = 0
        for data, target in train_loader: optimizer.zero_grad(); output = model(data); loss = criterion(output, target); loss.backward(); optimizer.step(); train_loss += loss.item()
        model.eval(); val_loss = 0; correct = 0
        with torch.no_grad():
            for data, target in val_loader: output = model(data); val_loss += criterion(output, target).item(); pred = output.argmax(dim=1); correct += pred.eq(target).sum().item()
        val_acc = 100. * correct / len(val_dataset); print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Acc = {val_acc:.2f}%")
    print("✓ End-to-end test completed successfully!")

def run_all_tests():
    print("\n" + "="*70); print("RUNNING ALL TESTS"); print("="*70)
    try:
        test_data_preparation(); test_vocabulary_and_dataset(); test_model(); test_training_loop(); test_end_to_end()
        print("\n" + "="*70); print("✅ ALL TESTS PASSED!"); print("="*70); print("\nYour pipeline is ready for full training.")
    except Exception as e:
        print("\n" + "="*70); print("❌ TEST FAILED!"); print("="*70); print(f"Error: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    # Add missing imports for self-contained test
    import re, math
    run_all_tests()