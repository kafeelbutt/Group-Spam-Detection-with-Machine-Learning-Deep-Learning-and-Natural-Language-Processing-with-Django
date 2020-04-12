from django.contrib import admin
from .models import Reviewer
from .models import Review
from .models import Product
# Register your models here.

admin.site.register(Review)
admin.site.register(Product)