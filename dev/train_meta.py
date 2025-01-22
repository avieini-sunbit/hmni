import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from hmni.input_helpers_torch import VocabularyProcessor
from hmni.siamese_network_torch import SiameseLSTM

def get_base_model_predictions(base_model, siamese_model, vocab_processor, data, device, batch_size=512):
    base_preds = []
    siamese_preds = []

    # Define feature names in the exact order they were used during training
    feature_names = [
        'partial', 'tkn_sort', 'tkn_set', 'sum_ipa',
        'pshp_soundex_first', 'iterativesubstring', 'bisim',
        'discountedlevenshtein', 'prefix', 'lcsstr', 'mlipns',
        'strcmp95', 'mra', 'editex', 'saps', 'flexmetric',
        'jaro', 'higueramico', 'sift4', 'eudex','aline', 'covington',
        'phoneticeditdistance'
    ]

    # Create mapping from data columns to model features

    print("Getting predictions...")
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    from tqdm.auto import tqdm

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        batch_data = data.iloc[start_idx:end_idx]
        features_df = batch_data[feature_names]
        # Get base model predictions for batch# features_df = pd.DataFrame(batch_data, columns=feature_names)
        batch_base_preds = base_model.predict_proba(features_df)[:, 1]
        base_preds.extend(batch_base_preds)

        # Prepare Siamese model inputs batch
        names_a = np.asarray(batch_data['name_a'].values)
        names_b = np.asarray(batch_data['name_b'].values)

        # Transform names to sequences
        x1_batch = np.asarray(list(vocab_processor.transform(names_a)))
        x2_batch = np.asarray(list(vocab_processor.transform(names_b)))

        # Convert to tensors
        x1_batch = torch.from_numpy(x1_batch).to(device)
        x2_batch = torch.from_numpy(x2_batch).to(device)

        # Get Siamese model predictions for batch
        with torch.no_grad():
            distances = siamese_model(x1_batch, x2_batch)
            similarities = 1 - distances.cpu().numpy()  # Convert distances to similarities
        siamese_preds.extend(similarities)

    return np.array(base_preds), np.array(siamese_preds)

def main():
    print("\n=== Phase 1: Loading Data ===")
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').iloc[:, 0]
    y_test = pd.read_csv('y_test.csv').iloc[:, 0]
    
    # Ensure name columns are strings
    X_train['name_a'] = X_train['name_a'].astype(str)
    X_train['name_b'] = X_train['name_b'].astype(str)
    X_test['name_a'] = X_test['name_a'].astype(str)
    X_test['name_b'] = X_test['name_b'].astype(str)
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    print("\n=== Phase 2: Loading Models ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model
    print("\nLoading Random Forest model...")
    base_model = joblib.load('../hmni/models/latin/base_model.pkl')
    
    # Load vocabulary and Siamese model
    print("\nLoading Siamese Network model...")
    vocab_processor = VocabularyProcessor.load('../hmni/models/latin/vocab_siamese.pkl')
    
    siamese_model = SiameseLSTM(
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=300,
        hidden_units=50,
        n_layers=2,
        dropout=0.2
    ).to(device)
    
    checkpoint = torch.load('../hmni/models/latin/siamese_model_final.pt', map_location=device)
    siamese_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n=== Phase 3: Getting Predictions ===")
    # Get predictions for training data
    train_base_pred, train_siamese_pred = get_base_model_predictions(
        base_model, siamese_model, vocab_processor, X_train, device
    )
    
    # Get predictions for test data
    test_base_pred, test_siamese_pred = get_base_model_predictions(
        base_model, siamese_model, vocab_processor, X_test, device
    )
    
    print("\n=== Phase 4: Training Meta Model ===")
    print("Preparing meta-features...")
    
    # Create meta-features DataFrame
    meta_train = pd.DataFrame({
        'predict_proba': train_base_pred,
        'siamese_sim': train_siamese_pred,
        'tkn_set': X_train['tkn_set'],
        'iterativesubstring': X_train['iterativesubstring'],
        'strcmp95': X_train['strcmp95']
    })
    
    meta_test = pd.DataFrame({
        'predict_proba': test_base_pred,
        'siamese_sim': test_siamese_pred,
        'tkn_set': X_test['tkn_set'],
        'iterativesubstring': X_test['iterativesubstring'],
        'strcmp95': X_test['strcmp95']
    })
    
    # Add interaction features
    meta_train['base_siamese_interaction'] = meta_train['predict_proba'] * meta_train['siamese_sim']
    meta_test['base_siamese_interaction'] = meta_test['predict_proba'] * meta_test['siamese_sim']
    
    # Train meta model with stronger regularization
    print("\nTraining meta model...")
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    
    # Use stronger regularization (smaller C value)
    base_clf = LogisticRegression(
        C=0.1,  # Stronger regularization
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000
    )
    
    # Use calibration to get better probability estimates
    meta_model = CalibratedClassifierCV(
        base_clf,
        cv=5,
        method='sigmoid'  # Platt scaling
    )
    
    meta_model.fit(meta_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_proba = meta_model.predict_proba(meta_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Evaluate
    print("\nMeta Model Evaluation:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(
        data=conf_matrix,
        columns=['Predicted: 0', 'Predicted: 1'],
        index=['Actual: 0', 'Actual: 1']
    ))
    
    # Save meta model
    print("\nSaving meta model...")
    joblib.dump(meta_model, '../hmni/models/latin/meta.pkl')
    print("\nTraining complete! Meta model saved as 'meta.pkl'")

if __name__ == "__main__":
    main() 