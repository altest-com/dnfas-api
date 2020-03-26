from typing import List

from django.db import models
from django.contrib.postgres.fields import JSONField


class Task(models.Model):

    TYPE_VIDEO_DETECT_FACES = 'video_detect_faces'
    TYPE_VIDEO_HUNT_FACES = 'video_hunt_faces'
    TYPE_VIDEO_DETECT_PERSON = 'video_detect_person'
    TYPE_VIDEO_HUNT_PERSON = 'video_hunt_person'
    TYPE_PREDICT_GENDERAGE = 'predict_genderage'
    TYPE_FACE_CLUSTERING = 'face_clustering'

    TYPE_CHOICES = [
        (TYPE_VIDEO_DETECT_FACES, 'video_detect_faces'),
        (TYPE_VIDEO_HUNT_FACES, 'video_hunt_faces'),
        (TYPE_VIDEO_DETECT_PERSON, 'video_detect_person'),
        (TYPE_VIDEO_HUNT_PERSON, 'video_hunt_person'),
        (TYPE_PREDICT_GENDERAGE, 'predict_genderage'),
        (TYPE_FACE_CLUSTERING, 'face_clustering')
    ]

    STATUS_CREATED = 'created'
    STATUS_RUNNING = 'running'
    STATUS_PAUSED = 'paused'
    STATUS_STOPPED = 'stopped'
    STATUS_KILLED = 'killed'
    STATUS_SUCCESS = 'success'
    STATUS_FAILURE = 'failure'
    STATUS_REVOKED = 'revoked'
    STATUS_RETRY = 'retry'

    STATUS_CHOICES = [
        (STATUS_CREATED, 'Created'),
        (STATUS_RUNNING, 'Running'),
        (STATUS_PAUSED, 'Paused'),
        (STATUS_STOPPED, 'Stopped'),
        (STATUS_KILLED, 'Killed'),
        (STATUS_SUCCESS, 'Success'),
        (STATUS_FAILURE, 'Failure'),
        (STATUS_REVOKED, 'Revoked'),
        (STATUS_RETRY, 'Retry')
    ]

    name = models.CharField(max_length=255, default='Task')
    tags = models.ManyToManyField(
        'Tag',
        related_name='tasks'
    )
    task_type = models.CharField(
        max_length=64,
        choices=TYPE_CHOICES,
        default=TYPE_VIDEO_DETECT_FACES
    )
    status = models.CharField(
        max_length=16,
        choices=STATUS_CHOICES,
        default=STATUS_CREATED
    )
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    schedule_start_at = models.DateTimeField(null=True, blank=True)
    schedule_stop_at = models.DateTimeField(null=True, blank=True)
    repeat_days = models.CharField(max_length=7, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    progress = models.FloatField(blank=True, default=-1)

    worker = models.ForeignKey(
        'Worker',
        null=True,
        blank=True,
        related_name='tasks',
        on_delete=models.SET_NULL
    )

    config = JSONField(blank=True, default=dict)

    info = JSONField(blank=True, default=dict)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class VTaskConfig:
    VIDEO_SOURCE_RECORD = 'record'
    VIDEO_SOURCE_CAMERA = 'camera'

    VIDEO_SOURCE_CHOICES = [
        (VIDEO_SOURCE_RECORD, 'record'),
        (VIDEO_SOURCE_CAMERA, 'camera')
    ]

    def __init__(self, *args, **kwargs):
        self.video_source_type: str = kwargs.get('video_source_type', '')
        self.video_source_id: int = kwargs.get('video_source_id', -1)
        self.start_at: float = kwargs.get('start_at', -1)
        self.stop_at: float = kwargs.get('stop_at', -1)


class VdfTaskConfig(VTaskConfig):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detection_min_height: int = kwargs.get('detection_min_height', 32)
        self.detection_min_score: float = kwargs.get('detection_min_score', 0.9)
        self.similarity_thresh: float = kwargs.get('similarity_thresh', 0.5)
        self.max_frame_size: int = kwargs.get('max_frame_size', 720)
        self.frontal_faces: bool = kwargs.get('frontal_faces', True)
        self.video_detect_interval: float = kwargs.get('video_detect_interval', 0.5)
        self.faces_time_memory: float = kwargs.get('faces_time_memory', 60)
        self.store_face_frames: bool = kwargs.get('store_face_frames', True)


class VhfTaskConfig(VdfTaskConfig):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hunted_subjects: List[str] = kwargs.get('hunted_subjects', [])


class PgaTaskConfig:

    def __init__(self, *args, **kwargs):
        self.min_created_at: float = kwargs.get('min_created_at', None)
        self.max_created_at: float = kwargs.get('min_created_at', None)
        self.overwrite: bool = kwargs.get('overwrite', False)


class PgaTaskInfo:

    def __init__(self):
        self.faces_count: int = 0
        self.processing_time: float = 0


class VTaskInfo:

    def __init__(self):
        self.frames_count: int = 0
        self.processing_time: float = 0
        self.frame_rate: float = 0


class FclTaskConfig:
    """Face clustering task config. """

    LINKAGE_WARD = 'ward'
    LINKAGE_AVERAGE = 'average'
    LINKAGE_SINGLE = 'single'
    LINKAGE_COMPLETE = 'complete'

    LINKAGE_CHOICES = [
        (LINKAGE_WARD, 'ward'),
        (LINKAGE_AVERAGE, 'average'),
        (LINKAGE_SINGLE, 'single'),
        (LINKAGE_COMPLETE, 'complete'),
    ]

    def __init__(self, *args, **kwargs):
        self.filter_back_weeks: int = kwargs.get('filter_back_weeks', None)
        self.filter_back_days: int = kwargs.get('filter_back_days', None)
        self.filter_back_hours: int = kwargs.get('filter_back_hours', None)
        self.filter_min_date: str = kwargs.get('filter_min_date', None)
        self.filter_max_date: str = kwargs.get('filter_max_date', None)
        self.filter_min_time: str = kwargs.get('filter_min_time', None)
        self.filter_max_time: str = kwargs.get('filter_max_time', None)
        self.filter_tasks: List[int] = kwargs.get('filter_tasks', [])
        self.filter_tasks_tags: List[int] = kwargs.get('filter_tasks_tags', [])
        self.top_dist_thr: float = kwargs.get('top_dist_thr', 0.6)
        self.low_dist_thr: float = kwargs.get('low_dist_thr', 0.6)
        self.edge_thr: float = kwargs.get('edge_thr', 0.5)
        self.linkage: str = kwargs.get('linkage', self.LINKAGE_WARD)
        self.memory_seconds: float = kwargs.get('memory_seconds', 3600)


class FclTaskInfo:
    """Face clustering task config. """

    def __init__(self, *args, **kwargs):
        self.processing_time: float = 0
        self.faces_count: float = 0


class HuntMatch(models.Model):
    target_subject = models.ForeignKey(
        'Subject',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='hunt_targets'
    )

    matched_subject = models.ForeignKey(
        'Subject',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='hunt_matches'
    )

    score = models.FloatField(blank=True, default=0)

    task = models.ForeignKey(
        'Task',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='hunts'
    )
