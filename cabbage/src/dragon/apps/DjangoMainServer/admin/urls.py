#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
