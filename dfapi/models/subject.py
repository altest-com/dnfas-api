import logging
from datetime import date
from datetime import datetime
from os import path
from typing import Tuple

import numpy as np
from dateutil.relativedelta import relativedelta
from django.conf import settings
from django.db import models
from django.db.models import QuerySet
from django.utils.functional import cached_property
from django.utils.timezone import make_aware

logger_name = settings.LOGGER_NAME
logger = logging.getLogger(logger_name)


class Subject(models.Model):

    SEX_MAN = 'man'
    SEX_WOMAN = 'woman'

    SEX_CHOICES = [
        (SEX_MAN, 'man'),
        (SEX_WOMAN, 'woman'),
    ]

    SKIN_WHITE = 'white'
    SKIN_BLACK = 'black'
    SKIN_BROWN = 'brown'

    SKIN_CHOICES = [
        (SKIN_WHITE, 'white'),
        (SKIN_BLACK, 'black'),
        (SKIN_BROWN, 'brown'),
    ]

    SOURCE_CAMERA = 'camera'
    SOURCE_VIDEO = 'video'
    SOURCE_MANUAL = 'manual'

    name = models.CharField(max_length=255, blank=True, default='')
    last_name = models.CharField(max_length=255, blank=True, default='')
    birthdate = models.DateField(blank=True, null=True)
    sex = models.CharField(max_length=16, choices=SEX_CHOICES, blank=True, default='')
    skin = models.CharField(max_length=16, choices=SKIN_CHOICES, blank=True, default='')
    pred_sex = models.CharField(max_length=16, choices=SEX_CHOICES, blank=True, default='')
    pred_age = models.PositiveIntegerField(blank=True, null=True)
    pred_sex_score = models.FloatField(
        blank=True,
        default=0.0
    )
    pred_age_var = models.FloatField(
        blank=True,
        default=0.0
    )

    # task = models.ForeignKey(
    #     'Task',
    #     null=True,
    #     blank=True,
    #     on_delete=models.CASCADE,
    #     related_name='subjects'
    # )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @cached_property
    def full_name(self):
        return f'{self.name} {self.last_name}'.strip()

    @cached_property
    def image(self):
        for face in self.faces.all():
            if face.image:
                return face.image
        return None

    @cached_property
    def timestamp(self):
        dt = None
        for face in self.faces.all():
            if dt is None or face.created_at > dt:
                dt = face.created_at

        return self.created_at if dt is None else dt

    @cached_property
    def age(self):
        if self.birthdate:
            return self.age_from_birthdate(self.birthdate)
        return None

    def __str__(self):
        return f'{self.full_name}' if self.full_name else f'[{self.pk}] unknown identity'

    class Meta:
        ordering = ['-created_at']

    @staticmethod
    def age_from_birthdate(birthdate):
        today = date.today()
        return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

    @staticmethod
    def birthdate_from_age(age):
        return datetime.now() - relativedelta(years=age)

    # @cached_property
    # def camera(self):
    #     if self.task.camera is not None:
    #         return self.task.camera.pk
    #     return None
    #
    # @cached_property
    # def video(self):
    #     if self.task is not None and self.task.video is not None:
    #         return self.task.video.pk
    #     return None

    @staticmethod
    def queryset_train_data(queryset: QuerySet) -> Tuple[np.ndarray, np.ndarray]:
        embeddings = []
        subjects = []
        queryset = queryset.exclude(faces__embeddings_bytes__isnull=True)
        for subject in queryset.iterator():
            for face in subject.faces.all():
                embeddings.append(face.embeddings)
                subjects.append(subject.id)

        embeddings = np.array(embeddings, np.float32)
        subjects = np.array(subjects, np.int32)

        return embeddings, subjects


# noinspection PyTypeChecker
class SubjectSegment(models.Model):

    NAMING_NAMED = 'named'
    NAMING_UNNAMED = 'unnamed'
    NAMING_ALL = ''

    NAMING_CHOICES = [
        (NAMING_NAMED, 'named'),
        (NAMING_UNNAMED, 'unnamed'),
        (NAMING_ALL, ''),
    ]

    disk_cached = models.BooleanField(
        blank=True,
        default=False,
        help_text='Indicates if linked train data if stored in disk.'
    )
    title = models.CharField(
        max_length=255,
        blank=True,
        help_text='The segment title.'
    )
    name = models.CharField(max_length=255, blank=True)
    naming = models.CharField(max_length=16, blank=True, choices=NAMING_CHOICES, default=NAMING_ALL)
    last_name = models.CharField(max_length=255, blank=True)
    min_birthdate = models.DateField(blank=True, null=True)
    max_birthdate = models.DateField(blank=True, null=True)
    min_timestamp = models.DateTimeField(blank=True, null=True)
    max_timestamp = models.DateTimeField(blank=True, null=True)
    sex = models.CharField(max_length=255, blank=True)
    skin = models.CharField(max_length=255, blank=True)
    count = models.IntegerField(default=0)

    cameras = models.ManyToManyField(
        'Camera',
        related_name='subject_segments'
    )

    videos = models.ManyToManyField(
        'VideoRecord',
        related_name='subject_segments'
    )

    tasks = models.ManyToManyField(
        'Task',
        related_name='subject_segments'
    )

    model_path = models.CharField(max_length=255, blank=True)

    updated_at = models.DateTimeField(null=True, blank=True)

    @cached_property
    def queryset(self) -> QuerySet:

        queryset = Subject.objects.all()

        if self.name:
            queryset = queryset.filter(name__icontains=self.name)

        if self.last_name:
            queryset = queryset.filter(last_name__icontains=self.last_name)

        if self.naming == self.NAMING_NAMED:
            queryset = queryset.exclude(
                name='', last_name=''
            )
        elif self.naming == self.NAMING_UNNAMED:
            queryset = queryset.filter(
                name='', last_name=''
            )

        tasks = []
        try:
            tasks = self.tasks.all()
        except ValueError:
            pass

        cameras = []
        try:
            cameras = self.cameras.all()
        except ValueError:
            pass

        videos = []
        try:
            videos = self.videos.all()
        except ValueError:
            pass

        if len(tasks):
            queryset = queryset.filter(task__in=tasks)
        elif len(cameras):
            queryset = queryset.filter(task__camera__in=cameras)
        elif len(videos):
            queryset = queryset.filter(task__video__in=videos)
        if self.min_timestamp is not None:
            queryset = queryset.filter(created_at__gt=self.min_timestamp)
        if self.max_timestamp is not None:
            queryset = queryset.filter(created_at__lt=self.max_timestamp)
        if self.min_birthdate is not None:
            queryset = queryset.filter(birthdate__gt=self.min_birthdate)
        if self.max_birthdate is not None:
            queryset = queryset.filter(birthdate__lt=self.max_birthdate)
        if self.sex:
            queryset = queryset.filter(sex=self.sex)
        if self.skin:
            queryset = queryset.filter(skin=self.skin)

        return queryset.distinct()

    def get_data(self):
        if not self.disk_cached or not self.model_path or self.is_outdated():
            embeddings, subjects = Subject.queryset_train_data(self.queryset)
            self.update_data(embeddings, subjects)
            return embeddings, subjects
        else:
            data = np.load(self.full_model_path)
            return data['embeddings'], data['subjects']

    def update_data(self, embeddings=None, subjects=None):
        if not self.disk_cached:
            return
        self.updated_at = make_aware(datetime.now())
        self.count = self.queryset.count()
        if self.model_path:
            if embeddings is None or subjects is None:
                embeddings, subjects = Subject.queryset_train_data(self.queryset)
            np.savez(self.full_model_path, subjects=subjects, embeddings=embeddings)
        self.save()

    def is_outdated(self):
        if not self.disk_cached or not path.exists(self.full_model_path):
            return True

        last_subject = None
        try:
            last_subject = self.queryset.all().latest('updated_at')
        except Subject.DoesNotExist:
            pass

        if (
            last_subject is None or
            self.updated_at is None or
            self.updated_at < last_subject.updated_at
        ):
            return True

        return False

    @property
    def full_model_path(self):
        return path.join(settings.DATA_ROOT, settings.MODELS_DATA_PATH, self.model_path)

    @property
    def min_age(self):
        return Subject.age_from_birthdate(self.min_birthdate)

    @min_age.setter
    def min_age(self, min_age):
        if min_age is not None:
            self.max_birthdate = Subject.birthdate_from_age(min_age)

    @property
    def max_age(self):
        return Subject.age_from_birthdate(self.max_birthdate)

    @max_age.setter
    def max_age(self, max_age):
        if max_age is not None:
            self.min_birthdate = Subject.birthdate_from_age(max_age)

