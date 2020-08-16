"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from . import views
from django.conf import settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns




urlpatterns = [
    # url(r'^admin/', (admin.site.urls)),
    # url('', views.homepageview, name='home'),
    # url('', views.servicespageview, name='services'),
    # url(r'^response/', views.response, name='response'),
    # url('', views.index2, name='index2'), # index url
    # url('', views.ajax_posting, name='ajax_posting'),
    # url('exists/', views.username_exists, name='exists'),
    # url('about/', views.about, name='about'),
    url(r'^$', views.index, name='index'),
    url(r'^services/', views.services, name='services'),
    url(r'^page3/', views.webpage3, name='webpage3'),
    url(r'^contact/', views.contact, name='contact'),
    url(r'^about/', views.about, name='about'),
    # url(r'^$', views.index, name='index'),
    # url(r'^services/', views.services, name='services'),
    # url(r'^portfolio/', views.portfolio, name='portfolio'),

]

if settings.DEBUG:
    urlpatterns += staticfiles_urlpatterns()
