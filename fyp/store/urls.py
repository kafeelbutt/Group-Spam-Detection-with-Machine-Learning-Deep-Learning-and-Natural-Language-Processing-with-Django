from django.urls import path
from . import views

urlpatterns= [
    path('index.html',views.index,name='index'),
    path('shop.html',views.shop,name='shop'),
    path('about.html',views.about,name='about'),
    path('contact.html',views.contact,name='contact'),
    path('cart.html',views.cart,name='cart'),
    path('product-detail.html',views.productdetail,name='product-detail'),
    path('about.html',views.about,name='about'),
    path('checkout.html',views.checkout,name='checkout'),
    path('order-complete.html',views.ordercomplete,name='order-complete'),
    path('add-to-wishlist.html',views.wishlist,name='order-add-to-wishlist'),
    path('register.html',views.register,name='reister'),
    path('login.html',views.login,name='login'),
    path('logout.html',views.logout,name='logout'),
]