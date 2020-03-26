from django.db import models
from django.conf import settings
from django.utils import timezone
import numpy as np


class Frame(models.Model):
    image = models.ImageField(upload_to=settings.FACES_IMAGES_PATH)
    timestamp = models.DateTimeField(null=True, blank=True)
    size_bytes = models.IntegerField(blank=True, default=0)


class Entity(models.Model):

    image = models.ImageField(
        upload_to=settings.FACES_IMAGES_PATH,
        null=True,
        blank=True
    )
    box_bytes = models.BinaryField(
        null=True,
        blank=True
    )
    embeddings_bytes = models.BinaryField(
        null=True,
        blank=True
    )
    size_bytes = models.IntegerField(
        blank=True,
        null=True
    )
    timestamp = models.DateTimeField(
        null=True,
        blank=True,
        default=timezone.now
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        null=True,
        blank=True
    )

    @property
    def box(self):
        if self.box_bytes is not None:
            return np.frombuffer(self.box_bytes, np.float32)
        return []

    @box.setter
    def box(self, val: [list, tuple, np.ndarray]):
        self.box_bytes = np.array(val, np.float32).tobytes()

    @property
    def embeddings(self):
        if self.embeddings_bytes is not None:
            return np.frombuffer(self.embeddings_bytes, np.float32)
        return []

    @embeddings.setter
    def embeddings(self, val: [list, np.ndarray]):
        self.embeddings_bytes = np.array(val, np.float32).tobytes()

    class Meta:
        abstract = True


class Face(Entity):

    SEX_MAN = 'man'
    SEX_WOMAN = 'woman'

    SEX_CHOICES = [
        (SEX_MAN, 'man'),
        (SEX_WOMAN, 'woman'),
    ]

    landmarks_bytes = models.BinaryField(
        null=True,
        blank=True
    )
    pred_sex = models.CharField(
        max_length=16,
        choices=SEX_CHOICES,
        blank=True,
        default=''
    )
    pred_sex_score = models.FloatField(
        blank=True,
        default=0.0
    )
    pred_age = models.PositiveIntegerField(
        blank=True,
        null=True
    )
    pred_age_var = models.FloatField(
        blank=True,
        default=0.0
    )
    frame = models.ForeignKey(
        'Frame',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='faces'
    )
    subject = models.ForeignKey(
        'Subject',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='faces'
    )
    task = models.ForeignKey(
        'Task',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='faces'
    )

    @property
    def landmarks(self):
        if self.landmarks_bytes is not None:
            return np.frombuffer(self.landmarks_bytes, np.float32).reshape((-1, 2))
        return []

    @landmarks.setter
    def landmarks(self, val: [list, np.ndarray]):
        self.landmarks_bytes = np.array(val, np.float32).tobytes()


class PersonBody(Entity):

    frame = models.ForeignKey(
        'Frame',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='person_bodies'
    )

    subject = models.ForeignKey(
        'Subject',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='person_bodies'
    )

    task = models.ForeignKey(
        'Task',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='person_bodies'
    )
