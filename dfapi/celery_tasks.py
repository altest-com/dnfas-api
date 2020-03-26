from datetime import timedelta, datetime

from celery import shared_task
from django.db.models import Count
from django.utils.timezone import make_aware

from dfapi.models import stat
from . import services
from .models import Face, Frame, Subject, Recognition
from .services.tasks import schedule_tasks, repeat_tasks


@shared_task
def update_hourly_stats():
    services.stats.update_time_stats(stat.Stat.RESOLUTION_HOUR)


@shared_task
def update_daily_stats():
    services.stats.update_time_stats(stat.Stat.RESOLUTION_DAY)


@shared_task
def control_tasks():
    schedule_tasks()
    repeat_tasks()


@shared_task
def clean_database():
    delete_orphan_faces = True
    delete_frames_without_faces = True
    delete_unnamed_subjects = True
    delete_unnamed_subjects_days = 365
    delete_recognitions = True
    delete_recognitions_days = 7

    now = make_aware(datetime.now())

    if delete_orphan_faces:
        Face.objects.filter(subject__isnull=True).delete()

    if delete_frames_without_faces:
        Frame.objects.annotate(
            faces_count=Count('faces')
        ).filter(
            faces_count=0
        ).delete()

    if delete_unnamed_subjects:
        max_timestamp = now - timedelta(days=delete_unnamed_subjects_days)
        Subject.objects.filter(
            name='',
            last_name='',
            updated_at__lt=max_timestamp
        ).delete()

    if delete_recognitions:
        max_timestamp = now - timedelta(days=delete_recognitions_days)
        Recognition.objects.filter(
            created_at__lt=max_timestamp
        ).delete()
