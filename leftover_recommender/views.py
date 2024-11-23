from functools import lru_cache

import numpy as np
from azure.storage.blob import BlobServiceClient
import pandas as pd
import json

from django.core.cache import cache
from rapidfuzz import process, fuzz
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import io
import ast
from django.conf import settings
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fine tune #1 to use memory cache
# Cache the DataFrame for 2 weeks
@lru_cache(maxsize=14)
def get_cached_dataframe():
    logger.info("Getting DataFrame from cache...")
    return get_dataframe_from_blob()

def get_dataframe_from_blob():
    """
    Fetches the dataset from Azure Blob Storage and loads it into a pandas DataFrame.
    """
    # Try to get from Django cache first
    cache_key = 'recipe_dataframe'
    df = cache.get(cache_key)

    if df is not None:
        logger.info("Retrieved DataFrame from Django cache")
        return df

    logger.info("Cache miss - fetching from Azure...")
    try:
        start_time = time.time()

        # Create a BlobServiceClient
        logger.info(f"Connecting to Azure Storage account: {settings.AZURE_STORAGE_ACCOUNT_NAME}")
        blob_service_client = BlobServiceClient(
            account_url=f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=settings.AZURE_STORAGE_ACCOUNT_KEY
        )

        # Get the container client
        logger.info(f"Accessing container: {settings.AZURE_STORAGE_CONTAINER_NAME}")
        container_client = blob_service_client.get_container_client(settings.AZURE_STORAGE_CONTAINER_NAME)

        # Download the blob content
        logger.info(f"Downloading blob: {settings.AZURE_BLOB_FILE_NAME}")
        blob_client = container_client.get_blob_client(settings.AZURE_BLOB_FILE_NAME)
        blob_data = blob_client.download_blob().readall().decode('utf-8')

        # Load into DataFrame
        logger.info("Converting blob data to DataFrame...")
        df = pd.read_csv(io.StringIO(blob_data))

        execution_time = round(time.time() - start_time, 2)
        logger.info(f"Successfully loaded DataFrame with {len(df)} rows in {execution_time} seconds")
        # Pre-process the DataFrame
        df['canonical_ingredients'] = df['canonical_ingredients'].apply(ast.literal_eval)

        # Fine tune #2 to use Djangho cache
        # Store in Django cache for 2 weeks
        cache.set(cache_key, df, timeout=1209600)

        logger.info(f"Successfully loaded and cached DataFrame with {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"Error fetching data from Azure Blob Storage: {str(e)}")
        raise Exception(f"Error fetching data from Azure Blob Storage: {e}")


# Finetune #3 to Vectorized matching for fast calculation in GPU
def find_matching_recipes_fuzzy(ingredients, df, threshold=70, match_threshold=0.8):
    """
    Optimized fuzzy matching function
    """
    logger.info(f"Starting optimized fuzzy matching with {len(ingredients)} ingredients...")

    def fuzzy_match_score(recipe_ingredients):
        if not recipe_ingredients:
            return 0

        # Vectorized matching
        scores = np.array([
            max((fuzz.ratio(ingredient, user_ingredient)
                 for user_ingredient in ingredients),
                default=0)
            for ingredient in recipe_ingredients
        ])

        matches = np.sum(scores >= threshold)
        return matches / len(recipe_ingredients) if recipe_ingredients else 0

    # Vectorized operations
    match_scores = df['canonical_ingredients'].apply(fuzzy_match_score)
    matching_mask = match_scores >= match_threshold

    matching_recipes = df[matching_mask].copy()
    matching_recipes['match_score'] = match_scores[matching_mask]
    matching_recipes = matching_recipes.sort_values('match_score', ascending=False)

    result = matching_recipes[['name', 'canonical_ingredients', 'match_score', 'steps']].to_dict(orient='records')

    # Convert canonical_ingredients back to list for JSON serialization
    for recipe in result:
        recipe['canonical_ingredients'] = list(recipe['canonical_ingredients'])

    return result

class TimerContext:
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = round(end_time - self.start_time, 2)
        logger.info(f"Completed {self.name} in {duration} seconds")
        return duration


@csrf_exempt
def recommend_recipes(request):
    """
    API endpoint to recommend recipes based on leftover ingredients.
    """
    request_start_time = time.time()
    logger.info("Received recipe recommendation request")
    timings = {}

    if request.method == 'POST':
        try:
            # Time request parsing
            with TimerContext('parse_request') as timer:
                data = json.loads(request.body)
                ingredients = data.get('ingredients', [])
                logger.info(f"Received ingredients: {ingredients}")
            timings['parse_request'] = timer.__exit__(None, None, None)

            if not isinstance(ingredients, list):
                logger.error("Invalid input: ingredients must be a list")
                return JsonResponse({'error': 'Ingredients must be a list'}, status=400)

            if not ingredients:
                logger.error("Invalid input: no ingredients provided")
                return JsonResponse({'error': 'No ingredients provided'}, status=400)

            # Time data fetching
            with TimerContext('fetch_data') as timer:
                df = get_dataframe_from_blob()
            timings['fetch_data'] = timer.__exit__(None, None, None)

            # Time recipe matching
            with TimerContext('find_recipes') as timer:
                recipes = find_matching_recipes_fuzzy(ingredients, df)
            timings['find_recipes'] = timer.__exit__(None, None, None)

            total_time = round(time.time() - request_start_time, 2)
            logger.info(f"Request completed successfully in {total_time} seconds")

            return JsonResponse({
                'recipes': recipes,
                'timings': timings,
                'total_time': sum(timings.values())
            }, status=200)

        except Exception as e:
            total_time = round(time.time() - request_start_time, 2)
            logger.error(f"Error processing request: {str(e)}")
            return JsonResponse({
                'error': str(e),
                'timings': timings,
                'total_time': sum(timings.values())
            }, status=500)

    logger.warning("Invalid request method received")
    return JsonResponse({'error': 'Invalid request method'}, status=405)