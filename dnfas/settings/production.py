from .base import *
import os

SPA_DIR = os.environ.get('DNFAS_SPA_DIR', '')

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ['DNFAS_SECRET_KEY']

DEBUG = False

MIDDLEWARE.remove('corsheaders.middleware.CorsMiddleware')

if SPA_DIR:
    SPA_DIR = os.path.realpath(SPA_DIR)
    TEMPLATES[0]['DIRS'] = [SPA_DIR]

    STATICFILES_DIRS = [
        os.path.join(SPA_DIR, 'static')
    ]

# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

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

# Django rest framework
REST_FRAMEWORK.update({
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'users.backends.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ]
})

# Enable CORS for specified domains:
CORS_ORIGIN_ALLOW_ALL = True
#
# CORS_ORIGIN_WHITELIST = (
#     'http://localhost',
# )
