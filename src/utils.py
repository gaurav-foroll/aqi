import os
import pickle

def checkCacheOrTrain(model_name, cache_path, train_func, *train_func_args):
    """
    A generic function to check if a model exists in cache. If not, it trains the model 
    using the provided training function and saves it to the cache.
    
    Args:
        model_name (str): The name of the model (used for cache file naming).
        cache_path (str): Path where the trained model will be cached.
        train_func (function): Function that trains the model.
        *train_func_args: Arguments required by the train_func for training the model.
    
    Returns:
        model: The trained or loaded model.
    """
    # Ensure the cache path exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Check if the model is already cached
    if os.path.exists(cache_path):
        print(f"Loading cached {model_name} model...")
        with open(cache_path, "rb") as f:
            model = pickle.load(f)
    else:
        print(f"Training new {model_name} model...")
        # Train the model using the provided training function and args
        model = train_func(*train_func_args)
        # Cache the trained model
        with open(cache_path, "wb") as f:
            pickle.dump(model, f)
    print(f"loaded {model_name} sucessfully")
    return model
