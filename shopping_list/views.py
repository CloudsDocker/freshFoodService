from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import ShoppingListItem
from .serializers import ShoppingListItemSerializer
from django.http import HttpResponse

class ShoppingListItemViewSet(viewsets.ModelViewSet):
    queryset = ShoppingListItem.objects.all()
    serializer_class = ShoppingListItemSerializer

    # Custom action to update only the quantity
    @action(detail=True, methods=['patch'])
    def update_quantity(self, request, pk=None):
        try:
            item = self.get_object()
            quantity = request.data.get('quantity')
            if quantity is not None:
                item.quantity = quantity
                item.save()
                return Response(self.serializer_class(item).data, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Quantity is required'}, status=status.HTTP_400_BAD_REQUEST)
        except ShoppingListItem.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

def homepage(request):
    return HttpResponse("Welcome to the Keep It Fresh API. Go to /api/shopping-list/ to see the shopping list.")


from django.http import HttpResponse

