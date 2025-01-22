import pandas as pd
import numpy as np
import requests
from hmni.matcher import Matcher
from tabulate import tabulate

def get_original_model_prediction(name_a, name_b):
    """Get prediction from original model API."""
    try:
        response = requests.post(
            'https://hmni-names-comparison.prod.sunbit.ai/api/v1.0/predictions',
            json={"data": {"ndarray": [0, 0], "names": [name_a, name_b]}}
        )
        if response.status_code == 200:
            return response.json()['data']['ndarray'][0]
        return np.nan
    except Exception as e:
        print(f"API Error: {str(e)}")
        return np.nan

def compare_names(name1, name2):
    # Create request payload matching the API format
    payload = {
        "data": {
            "ndarray": [0, 0],  # Placeholder array as per API format
            "names": [name1, name2]
        }
    }
    
    # Get predictions from our matcher
    matcher = Matcher()
    prediction = matcher.similarity(name1, name2)
    
    # Format results
    return {
        'base_model': matcher.base_model_inf(matcher.get_base_features(name1, name2)),
        'siamese_nn': matcher.siamese_inf((name1, name2)),
        'meta_model': matcher.meta_inf(name1, name2),
        'final_score': prediction,
        'original_api': None,  # This would be filled by actual API call
        'difference': None  # This would be calculated with actual API response
    }

def compare_predictions(test_cases):
    """Compare predictions between our model and original API."""
    print("Initializing matcher...")
    matcher = Matcher(model='latin')
    
    results = []
    print("\nProcessing test cases...")
    
    for name_a, name_b in test_cases:
        # Get predictions from each component
        base_features = matcher.get_base_features(name_a, name_b)
        base_pred = matcher.base_model_inf(base_features)
        siamese_pred = matcher.siamese_inf((name_a, name_b))
        meta_pred = matcher.meta_inf(name_a, name_b)
        final_pred = matcher.similarity(name_a,name_b)
        # Get final prediction with lastname check

            
        orig_pred = get_original_model_prediction(name_a, name_b)
        
        results.append({
            'Name A': name_a,
            'Name B': name_b,
            'Base Model': f"{base_pred:.4f}",
            'Siamese NN': f"{siamese_pred:.4f}",
            'Meta Model': f"{meta_pred:.4f}",
            'Final Score': f"{final_pred:.4f}",
            'Original API': f"{orig_pred:.4f}" if not np.isnan(orig_pred) else "N/A",
            'Difference': f"{abs(final_pred - orig_pred):.4f}" if not np.isnan(orig_pred) else "N/A"
        })
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(results)
    
    # Print results table
    print("\nDetailed Comparison Results:")
    print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
    
    # Calculate summary statistics
    valid_diffs = [float(x) for x in df['Difference'] if x != "N/A"]
    if valid_diffs:
        print(f"\nSummary Statistics:")
        print(f"Average Difference: {np.mean(valid_diffs):.4f}")
        print(f"Maximum Difference: {np.max(valid_diffs):.4f}")
    
    # Model Analysis
    print("\nModel Analysis:")
    print("1. Base Model:")
    base_scores = [float(x) for x in df['Base Model']]
    print(f"   - Average Score: {np.mean(base_scores):.4f}")
    print(f"   - Score Range: {np.min(base_scores):.4f} to {np.max(base_scores):.4f}")
    
    print("\n2. Siamese Neural Network:")
    siamese_scores = [float(x) for x in df['Siamese NN']]
    print(f"   - Average Score: {np.mean(siamese_scores):.4f}")
    print(f"   - Score Range: {np.min(siamese_scores):.4f} to {np.max(siamese_scores):.4f}")
    
    print("\n3. Meta Model:")
    meta_scores = [float(x) for x in df['Meta Model']]
    print(f"   - Average Score: {np.mean(meta_scores):.4f}")
    print(f"   - Score Range: {np.min(meta_scores):.4f} to {np.max(meta_scores):.4f}")

def main():
    # Define test cases
    test_cases = [
        # Exact matches
        ("John", "John"),
        ("Michael", "Michael"),
        ("Elizabeth", "Elizabeth"),


        # Similar first names - loose matches
        ("Avi", "Avraham"),
        ("Edo", "Ido"),
        ("Philip", "Filip"),
        ("Robert", "Roberto"),
        ("Chris", "Christopher"),
        ("Mike", "Michael"),
        ("Dave", "David"),
        ("Tony", "Anthony"),
        ("Bill", "William"),
        ("Dick", "Richard"),
        #
        # Similar first names - tighter matches
        ("Steven", "Stephen"),
        ("Sara", "Sarah"),
        ("Daniel", "Danielle"),
        ("Eric", "Erik"),
        ("Brian", "Bryan"),
        ("Sean", "Shaun"),

        # Full names with matching last names
        ("Edo Israeli", "Ido Israeli"),
        ("Philip Smith", "Filip Smith"),
        ("Michael Brown", "Mike Brown"),
        ("Robert Johnson", "Bob Johnson"),
        ("William Wilson", "Bill Wilson"),
        ("Christopher Lee", "Chris Lee"),
        ("Elizabeth Taylor", "Liz Taylor"),
        ("Katherine Johnson", "Katie Johnson"),

        # # Full names with different last names
        # ("Avi Cohen", "Avraham Katz"),
        # ("Philip Smith", "Filip Johnson"),
        # ("Michael Brown", "Mikhail Smith"),
        # ("Robert Wilson", "Bob Thompson"),
        # ("William Clark", "Bill Rodriguez"),
        # ("Christopher Lee", "Chris Anderson"),
        #
        # # Traditional variations and cultural equivalents
        # ("Yosef", "Joseph"),
        # ("Jacob", "Yaakov"),
        # ("Alexander", "Sasha"),
        # ("John", "Juan"),
        # ("James", "Santiago"),
        # ("Mary", "Maria"),
        # ("Peter", "Pedro"),
        # ("George", "Jorge"),
        # ("Michael", "Miguel"),
        # ("Nicholas", "Nicolas"),
        #
        # # Complex cultural variations
        # ("Mohammed", "Muhammad"),
        # ("Yohanan", "Johannes"),
        # ("Miriam", "Mary"),
        # ("Yeshua", "Jesus"),
        # ("Chaim", "Hayim"),
        # ("Moshe", "Moses"),
        # ("Yitzchak", "Isaac"),
        # ("Avraham", "Ibrahim"),
        #
        # # Names with prefixes/suffixes
        # ("Dr. John Smith", "John Smith"),
        # ("Mr. Robert Brown", "Robert Brown"),
        # ("James Wilson Jr.", "James Wilson"),
        # ("William Johnson III", "William Johnson"),
        # ("Mrs. Sarah Davis", "Sarah Davis"),
        #
        # # Compound names
        # ("Jean-Pierre", "Jean Pierre"),
        # ("Mary Jane", "Mary-Jane"),
        # ("Anna Maria", "Anna-Maria"),
        # ("Jean Michel", "Jean-Michel"),
        #
        # # Different names (hard negatives)
        # ("John", "David"),
        # ("Sarah", "Michael"),
        # ("Daniel", "Elizabeth"),
        # ("Robert", "William"),
        # ("Thomas", "Richard"),
        # ("Margaret", "Elizabeth"),
        #
        # # Challenging cases
        # ("J. Robert", "Bob"),
        # ("Peggy", "Margaret"),
        # ("Jack", "John"),
        # ("Chuck", "Charles"),
        # ("Beth", "Elizabeth"),
        # ("Jim", "James"),
        #
        # # Names with typos or common misspellings
        # ("Catherine", "Katherine"),
        # ("Kristin", "Kristen"),
        # ("Deborah", "Debra"),
        # ("Jeffrey", "Geoffrey"),
        # ("Michele", "Michelle"),
        # ("Phillip", "Philip"),
        #
        # # International variations
        # ("José", "Joseph"),
        # ("André", "Andrew"),
        # ("François", "Francis"),
        # ("Mikhail", "Michael"),
        # ("Giovanni", "John"),
        # ("Carlos", "Charles"),
        # ("Pavel", "Paul"),
        # #
        # # # Complex last name variations
        # ("David McDonald", "David MacDonald"),
        # ("Sarah Meyer", "Sarah Meier"),
        # ("John O'Brien", "John OBrien"),
        # ("Anna De Silva", "Anna DaSilva"),
        # ("Robert MacKenzie", "Robert McKenzie"),
        #
        # # # Multiple word last names
        # ("James Van Der Beek", "James VanDerBeek"),
        # ("Mary St. John", "Mary Saint John"),
        # ("Robert De La Rosa", "Robert Delarosa"),
        # ("William Van Dyke", "William VanDyke")
    ]
    
    compare_predictions(test_cases)

if __name__ == "__main__":
    main() 