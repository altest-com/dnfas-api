from django.db import models


class RecognitionMatch(models.Model):

    score = models.FloatField(null=True, blank=True)

    subject = models.ForeignKey(
        'Subject',
        on_delete=models.CASCADE,
        related_name='recognition_matches'
    )

    recognition = models.ForeignKey(
        'Recognition',
        on_delete=models.CASCADE,
        related_name='matches'
    )


class Recognition(models.Model):

    sim_thresh = models.FloatField(blank=True, default=0.5)
    max_matches = models.IntegerField(blank=True, default=5)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    face = models.ForeignKey(
        'Face',
        on_delete=models.CASCADE,
        related_name='recognitions'
    )

    segments = models.ManyToManyField(
        'SubjectSegment',
        related_name='recognitions'
    )

    # filter = models.ForeignKey(
    #     'SubjectSegment',
    #     on_delete=models.CASCADE,
    #     related_name='filters',
    #     null=True,
    #     blank=True
    # )

    def __str__(self):
        return f'Recognition <{self.pk}>'

    class Meta:
        ordering = ['-created_at']
