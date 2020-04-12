from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User,auth
from .models import Review
from .models import Product
import datetime
from . import model
import itertools
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.externals import joblib
# Create your views here.

def index(request):
    return render(request,'index.html')

def shop(request):
    if request.method == 'POST':
        range1=request.POST.get('range1','0')
        range1=int(range1)
        range2=request.POST.get('range2','0')
        range1=int(range1)
        prod=Product.objects.filter(Price__gte=range1,Price__lte=range2)
        parameter='None'
        return render(request,'shop.html',{'prods':prod,'parameter':parameter})
    else:
        parameter = request.GET.get('Categ','None')
        prod=Product.objects.all()
        cate=Product.objects.values('Category').distinct()
        if parameter!= 'None':
            return render(request,'shop.html',{'prods':prod,'cates':cate,'parameter':parameter})
        return render(request,'shop.html',{'prods':prod,'cates':cate,'parameter':parameter})

def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'contact.html')

def cart(request):
    return render(request,'cart.html')

def productdetail(request):
    if request.method == 'POST':
        ReviewField=request.POST.get('ReviewField1','')
        Rating=request.POST.get('rate','0')
        parameter=request.POST.get('para','0')
        parameter=int(parameter)       
        filename = "./demo_model.pickle"
        #m = model.Model("./raw_data_back_model.csv")
        #m.save(filename)
        loaded_model = model.Model.load(filename)
        
        s=loaded_model.predict(R_id=int(request.user.id),p_id=414,rating=float(Rating),label=-1,date=str("2012-09-23"))
        print(s)
        #print(loaded_model.final_groups.final_df)
        with open('./tf-idf.pkl','rb') as f:
            cv=pickle.load(f)
        filename="./random_forest.sav"
        claasifier_rf=joblib.load(filename)
        data = {'comment_text':[ReviewField]}
        feature_vector = pd.DataFrame(data)
        print("Feature Vector before encoding:" )
        print(feature_vector)

        lemmatizer = WordNetLemmatizer()
        corpus = []
        review = re.sub('[^a-zA-Z]', ' ', feature_vector['comment_text'][0])
        review = review.lower()
        review = review.split()
            
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
        feature_vectors_enc = cv.transform(corpus)
        df2=pd.DataFrame(feature_vectors_enc.toarray(),columns=cv.get_feature_names())
        print(df2)
        result=claasifier_rf.predict(df2)
        print(result)
        if result==0:
            st="Not Spam"
        else:
            st="Spam"
        
        store_Review=Review(ReviewerID_id=request.user.id,ReviewDesc=ReviewField,prod_id=parameter,rating=Rating,date=datetime.date.today(),Spam=st)
        store_Review.save()
        
        return redirect("product-detail.html?product=%d"%parameter)
    else:
        parameter = request.GET.get('product','0')
        parameter=int(parameter)
        rev=reversed(Review.objects.all())
        prod=Product.objects.get(id=parameter)
        prods=Product.objects.all()
        Count = Review.objects.filter(prod_id=parameter).count()
        return render(request,'product-detail.html',{'Count':Count,'prods':prods,'prod':prod,'parameter':parameter,'revs':rev})

def checkout(request):
    return render(request,'checkout.html')

def ordercomplete(request):
    return render(request,'order-complete.html')

def wishlist(request):
    return render(request,'add-to-wishlist.html')

def register(request):
    if request.method == 'POST':
        name= request.POST.get('name'," ")
        email= request.POST.get('email',' ')
        password1= request.POST.get('password1',' ')
        password2= request.POST.get('password2',' ')

        if password1 == password2:
            if User.objects.filter(username=name).exists():
                messages.info(request,'Username Taken')
                return redirect('register.html')
            elif User.objects.filter(email=email).exists():
                messages.info(request,"email taken")
                return redirect('register.html')
            else:
                user=User.objects.create_user(username=name,password=password1,email=email)
                user.save()
                return redirect("login.html")
        else:
            messages.info(request,"password not matching")
            return redirect('register.html')
        return redirect('index.html')
    else:
        return render(request,'register.html')

def login(request):
    if request.method=='POST':
        username=request.POST.get("username")
        password=request.POST.get("password")
        user=auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            return redirect("index.html")
        else:
            messages.info(request,'invalid credentials')
            return redirect("login.html")
    else:
        return render(request,'login.html')     
def logout(request):
    auth.logout(request)
    return redirect('index.html')
