import logging
import os
import uuid
from os import path

from django.conf import settings
from django.db.models.signals import post_save, pre_save, post_delete
from django.dispatch import receiver

from . import services
from .models import (
    Face,
    Frame,
    VideoRecord,
    SubjectSegment,
    Task,
    Notification
)
from .services.faces import face_analyzer

logger_name = settings.LOGGER_NAME
logger = logging.getLogger(logger_name)


@receiver(post_delete, sender=Face)
def delete_face_image_on_delete(sender, instance, **kwargs):
    if instance.image:
        if os.path.isfile(instance.image.path):
            os.remove(instance.image.path)


@receiver(pre_save, sender=Face)
def delete_face_image_on_change(sender, instance: Face, **kwargs):
    if not instance.pk:
        return False

    try:
        old_file = Face.objects.get(pk=instance.pk).image
    except Face.DoesNotExist:
        return False

    new_file = instance.image
    if not old_file == new_file:
        instance.box_bytes = None
        instance.embeddings_bytes = None
        instance.landmarks_bytes = None
        if os.path.isfile(old_file.path):
            os.remove(old_file.path)


# @receiver(post_save, sender=Face)
# def on_face_post_save(sender, instance: Face, **kwargs):
#     if not instance:
#         return
#
#     if hasattr(instance, 'dirty'):
#         del instance.dirty
#         return
#
#     if instance.size_bytes is None:
#         instance.dirty = True
#         instance.size_bytes = instance.image.size
#         instance.save()
#
#     if instance.frame is not None and instance.frame.size_bytes == 0:
#         instance.frame.size_bytes = instance.frame.image.size
#         instance.frame.save()
#
#     if (
#         instance.embeddings_bytes is None or
#         instance.landmarks_bytes is None or
#         instance.box_bytes is None
#     ):
#         if os.path.isfile(instance.image.path):
#             face_analyzer.analyze_face(instance.pk)
#             # instance = detect_face(instance)


@receiver(post_delete, sender=Frame)
def delete_frame_image_on_delete(sender, instance: Frame, **kwargs):
    if instance.image:
        if os.path.isfile(instance.image.path):
            os.remove(instance.image.path)


@receiver(pre_save, sender=Frame)
def auto_delete_file_on_change(sender, instance: Frame, **kwargs):
    if not instance.pk:
        return False

    try:
        old_file = Frame.objects.get(pk=instance.pk).image
    except Frame.DoesNotExist:
        return False

    new_file = instance.image
    if not old_file == new_file:
        if os.path.isfile(old_file.path):
            os.remove(old_file.path)


@receiver(pre_save, sender=VideoRecord)
def video_record_pre_save(sender, instance: VideoRecord = None, **kwargs):
    if instance is None:
        return
    if not instance.name:
        instance.name = path.basename(instance.path)
    services.media.fill_video(instance)


@receiver(post_save, sender=VideoRecord)
def video_record_post_save(sender, instance: VideoRecord = None, **kwargs):
    if instance is None:
        return
    services.media.create_video_thumbs(instance)


@receiver(pre_save, sender=SubjectSegment)
def subject_segment_pre_save(sender, instance: SubjectSegment = None, **kwargs):
    if instance is not None and instance.disk_cached and not instance.model_path:
        instance.model_path = f'segment_{str(uuid.uuid4())}.npz'


@receiver(post_save, sender=SubjectSegment)
def subject_segment_post_save(sender, instance: SubjectSegment = None, **kwargs):
    if instance is None or not instance.disk_cached:
        return

    if hasattr(instance, 'dirty'):
        del instance.dirty
        return

    instance.dirty = True
    if instance.is_outdated():
        instance.update_data()


@receiver(post_delete, sender=SubjectSegment)
def delete_frame_image_on_delete(sender, instance: SubjectSegment, **kwargs):
    if instance is not None and os.path.isfile(instance.full_model_path):
        os.remove(instance.full_model_path)


@receiver(post_delete, sender=Task)
def task_post_delete(sender, instance: Task, **kwargs):
    if instance:
        Notification.objects.filter(resource=instance.pk).delete()
