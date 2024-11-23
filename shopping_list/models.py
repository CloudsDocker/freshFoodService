from django.db import models

class ShoppingListItem(models.Model):
    name = models.CharField(max_length=100)
    quantity = models.IntegerField()
    purchased = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.name} ({self.quantity})"
from django.db import models

# Create your models here.
