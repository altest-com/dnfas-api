from django.db import models


class Stat(models.Model):

    RESOLUTION_HOUR = 'hour'
    RESOLUTION_DAY = 'day'
    RESOLUTION_ALL = 'all'

    RESOLUTION_CHOICES = [
        (RESOLUTION_HOUR, 'hour'),
        (RESOLUTION_DAY, 'day'),
        (RESOLUTION_ALL, 'all')
    ]

    name = models.CharField(max_length=255)
    timestamp = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    value = models.FloatField(default=0)
    resolution = models.CharField(max_length=16, choices=RESOLUTION_CHOICES)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f'Stat: ({self.name}, {self.value}, {self.timestamp}, {self.resolution})'
