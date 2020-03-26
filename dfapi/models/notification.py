from django.db import models


class Notification(models.Model):

    CATEGORY_TASK = 'task'

    CATEGORY_CHOICES = [
        (CATEGORY_TASK, 'Task')
    ]

    DTYPE_ERROR = 'error'
    DTYPE_WARN = 'warn'
    DTYPE_INFO = 'info'

    DTYPE_CHOICES = [
        (DTYPE_ERROR, 'Task'),
        (DTYPE_WARN, 'Warn'),
        (DTYPE_INFO, 'Info'),
    ]

    category = models.CharField(
        max_length=16,
        choices=CATEGORY_CHOICES
    )
    dtype = models.CharField(
        max_length=16,
        choices=DTYPE_CHOICES
    )
    title = models.CharField(max_length=255)
    message = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    resource = models.IntegerField()
    seen = models.BooleanField(default=False)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f'[{self.pk}] {self.message}'
