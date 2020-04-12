from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Review(models.Model):
    ReviewerID=models.ForeignKey(User,on_delete=models.CASCADE)
    date = models.CharField(max_length=50)
    prod_id= models.IntegerField()
    rating= models.IntegerField()
    ReviewDesc = models.CharField(max_length=1000)
    Spam=models.CharField(max_length=1000)
    
class Reviewer(models.Model):
    Name = models.CharField(max_length=100)
    Email = models.CharField(max_length=100)
    Password = models.CharField(max_length=100)

class Product(models.Model):
    id = models.IntegerField(primary_key=True)
    ProductName = models.CharField(max_length=100)
    Category = models.CharField(max_length=100)
    Price = models.IntegerField()
    img = models.ImageField(upload_to='pics')
    Description=models.CharField(max_length=300)