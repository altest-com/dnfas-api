from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

_hosts = os.environ.get('DNFAS_ALLOWED_HOSTS', None)

if _hosts is not None:
    ALLOWED_HOSTS.extend(_hosts.split(','))


# Django rest framework
REST_FRAMEWORK.update({
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'users.backends.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ]
})

# Enable CORS for all domains
CORS_ORIGIN_ALLOW_ALL = True