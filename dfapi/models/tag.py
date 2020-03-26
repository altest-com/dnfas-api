from django.db import models


class Tag(models.Model):
    MODEL_TASK = 'task'
    MODEL_CAMERA = 'camera'
    MODEL_VIDEO = 'video'

    MODEL_CHOICES = [
        (MODEL_TASK, 'task'),
        (MODEL_CAMERA, 'task'),
        (MODEL_VIDEO, 'video')
    ]

    name = models.CharField(max_length=64)
    model = models.CharField(max_length=64, choices=MODEL_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


