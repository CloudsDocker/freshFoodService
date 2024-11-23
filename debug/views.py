import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def debug_env_vars(request):
    """
    Debug view to list all environment variables.
    """
    env_vars = {key: value for key, value in os.environ.items()}
    return JsonResponse(env_vars, safe=False)
