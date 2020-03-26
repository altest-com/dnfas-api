from django.db import models
from django.conf import settings


class Worker(models.Model):

    name = models.CharField(max_length=255, blank=True)
    api_url = models.URLField()
    username = models.CharField(max_length=255, blank=True)
    password = models.CharField(max_length=255, blank=True)
    max_load = models.IntegerField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def is_self(self):
        return self.name.lower() == settings.WORKER_NAME.lower()

    class Meta:
        ordering = ['updated_at']

    def __str__(self):
        return str(self.api_url)
