import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from clarifai.client.model import Model
import tempfile

# Replace these with your actual Clarifai credentials
PAT = "d7d06212c0da4f6eb953cd9ade5f8fd5"  # Personal Access Token
MODEL_URL = "https://clarifai.com/clarifai/main/models/food-item-recognition"

@csrf_exempt
def analyze_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']

        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        try:
            # Initialize the Clarifai model
            model = Model(url=MODEL_URL, pat=PAT)

            # Use the model to predict concepts from the image file
            model_prediction = model.predict_by_filepath(temp_file_path, input_type="image")

            # Check if predictions were returned
            if model_prediction.outputs:
                concepts = model_prediction.outputs[0].data.concepts
                # Extract only ingredient names
                ingredients = [concept.name for concept in concepts]
                return JsonResponse({'ingredients': ingredients})
            else:
                return JsonResponse({'error': 'No predictions found'}, status=500)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)

    return JsonResponse({'error': 'Invalid request'}, status=400)
