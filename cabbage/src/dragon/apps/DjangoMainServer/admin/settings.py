#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
import os
import dotenv

from django.core.management.utils import get_random_secret_key


dotenv.load_dotenv('.env')

config = {
    'SECRET': os.environ.get('SECRET', 'dragon-secret-key-' + get_random_secret_key()),
    'DEBUG': os.environ.get('DEBUG', True),
    'LANGUAGE_CODE': os.environ.get('LANGUAGE_CODE', 'en-us'),
    'TIMEZONE': os.environ.get('TIMEZONE', 'UTC'),
}

SECRET_KEY = config.get('SECRET')
DEBUG = config.get('DEBUG')
ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0',
    '::1',
    '::',
]

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'apps.DjangoMainServer.admin.urls'

TEMPLATES = [
    {
        'APP_DIRS': True,
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'apps.DjangoMainServer.admin.wsgi.application'
ASGI_APPLICATION = 'apps.DjangoMainServer.admin.asgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = config.get('LANGUAGE_CODE')
TIME_ZONE = config.get('TIMEZONE')
USE_I18N = True
USE_TZ = True
STATIC_URL = 'static/'
STATICFILES_DIRS = [
    'static',
]
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
