from azure.storage.blob import BlobServiceClient
import pandas as pd
import json
from rapidfuzz import process, fuzz
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import io
import ast
from django.conf import settings


def get_dataframe_from_blob():
    """
    Fetches the dataset from Azure Blob Storage and loads it into a pandas DataFrame.
    """
    try:
        # Create a BlobServiceClient
        blob_service_client = BlobServiceClient(
            account_url=f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=settings.AZURE_STORAGE_ACCOUNT_KEY
        )
        # Get the container client
        container_client = blob_service_client.get_container_client(settings.AZURE_STORAGE_CONTAINER_NAME)
        # Download the blob content as a stream
        blob_client = container_client.get_blob_client(settings.AZURE_BLOB_FILE_NAME)
        blob_data = blob_client.download_blob().readall().decode('utf-8')  # Decode binary content to string
        # Load the content into a pandas DataFrame
        df = pd.read_csv(io.StringIO(blob_data))
        return df
    except Exception as e:
        raise Exception(f"Error fetching data from Azure Blob Storage: {e}")


def find_matching_recipes_fuzzy(ingredients, df, threshold=70, match_threshold=0.8):
    """
    Fuzzy match the provided ingredients with recipes in the dataset.
    """
    def fuzzy_match_score(recipe_ingredients):
        matches = 0
        # Ensure recipe_ingredients is a sequence
        if not isinstance(recipe_ingredients, (list, tuple)):
            recipe_ingredients = []
        for recipe_ingredient in recipe_ingredients:
            result = process.extractOne(recipe_ingredient, ingredients, scorer=fuzz.ratio)
            if result and result[1] >= threshold:
                matches += 1
        return matches / len(recipe_ingredients) if recipe_ingredients else 0

    df['match_score'] = df['canonical_ingredients'].apply(
        lambda x: fuzzy_match_score(ast.literal_eval(x))  # Safely parse stringified lists
    )
    matching_recipes = df[df['match_score'] >= match_threshold].sort_values(by='match_score', ascending=False)
    return matching_recipes[['name', 'canonical_ingredients', 'match_score', 'steps']].to_dict(orient='records')


import time
from functools import wraps


class TimerContext:
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = round(end_time - self.start_time, 2)
        return duration


@csrf_exempt
def recommend_recipes(request):
    """
    API endpoint to recommend recipes based on leftover ingredients.
    """
    timings = {}

    if request.method == 'POST':
        try:
            # Time request parsing
            with TimerContext('parse_request') as timer:
                data = json.loads(request.body)
                ingredients = data.get('ingredients', [])
            timings['parse_request'] = timer.__exit__(None, None, None)

            if not isinstance(ingredients, list):
                return JsonResponse({'error': 'Ingredients must be a list'}, status=400)

            if not ingredients:
                return JsonResponse({'error': 'No ingredients provided'}, status=400)

            # Time data fetching
            with TimerContext('fetch_data') as timer:
                df = get_dataframe_from_blob()
            timings['fetch_data'] = timer.__exit__(None, None, None)

            # Time recipe matching
            with TimerContext('find_recipes') as timer:
                recipes = find_matching_recipes_fuzzy(ingredients, df)
            timings['find_recipes'] = timer.__exit__(None, None, None)

            return JsonResponse({
                'recipes': recipes,
                'timings': timings,
                'total_time': sum(timings.values())
            }, status=200)

        except Exception as e:
            return JsonResponse({
                'error': str(e),
                'timings': timings,
                'total_time': sum(timings.values())
            }, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)