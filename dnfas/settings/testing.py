from .development import *
import os


# Database
DATABASES['default']['TEST'] = {
    'NAME': os.environ['DNFAS_DB_NAME']
}

MEDIA_ROOT = os.path.realpath(os.path.join(BASE_DIR, 'storage/testing'))
DATA_ROOT = MEDIA_ROOT

MEDIA_PATHS = [
    VIDEO_RECORDS_PATH,
    VIDEO_THUMBS_PATH,
    FACES_IMAGES_PATH,
    MODELS_DATA_PATH
]

for media_path in MEDIA_PATHS:
    full_path = os.path.join(MEDIA_ROOT, media_path)
    os.makedirs(full_path, exist_ok=True)
