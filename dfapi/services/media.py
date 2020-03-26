from django.core.exceptions import ObjectDoesNotExist, ValidationError

from os import path
from glob import iglob
from datetime import datetime

from django.utils.timezone import make_aware
from django.conf import settings

import cv2 as cv

from ..models import VideoRecord, VideoThumb
from cvtlib.video import VideoCapture

FILENAME_DATE_SEPARATOR = '__'
FILENAME_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'


def is_valid_video_path(rel_path):

    try:
        VideoRecord.objects.get(path=rel_path)
        msg = f'A video with the path "{rel_path}" already exists.'
        return False, msg
    except ObjectDoesNotExist:
        pass

    full_path = path.join(settings.MEDIA_ROOT, settings.VIDEO_RECORDS_PATH, rel_path)

    if not path.isfile(full_path):
        msg = f'File "{full_path}" does not exists.'
        return False, msg

    video_capture = VideoCapture(full_path)
    video_capture.open()
    if not video_capture.is_opened():
        msg = f'Video file "{full_path}" can not be opened.'
        return False, msg
    video_capture.release()

    return True, ''


def load_videos(dir_path):

    root_dir = path.join(settings.MEDIA_ROOT, settings.VIDEO_RECORDS_PATH)
    file_paths = []
    for ext in settings.VIDEO_SUPPORTED_EXT:
        root_path_ext = path.join(root_dir, dir_path, f'**/*.{ext}')
        file_paths.extend(iglob(root_path_ext, recursive=True))

    for file_path in file_paths:
        rel_path = path.relpath(file_path, root_dir)
        valid, msg = is_valid_video_path(rel_path)
        if valid:
            VideoRecord.objects.create(path=rel_path)


def fill_video(instance: VideoRecord):

    valid, msg = is_valid_video_path(instance.path)
    if not valid:
        raise ValidationError(msg)

    full_path = instance.full_path
    video_capture = VideoCapture(full_path)
    video_capture.open()
    if video_capture.is_opened():
        starts_at, finish_at = None, None
        filename = path.splitext(path.basename(instance.path))[0]
        dates = filename.split(FILENAME_DATE_SEPARATOR)
        if len(dates) == 2:
            try:
                starts_at = make_aware(datetime.strptime(dates[0], FILENAME_DATE_FORMAT))
                finish_at = make_aware(datetime.strptime(dates[1], FILENAME_DATE_FORMAT))
            except ValueError:
                print(f'Unable to parse datetime info from video file "{full_path}".')

        frame_width, frame_height = 0, 0
        duration_seconds = 0
        size_bytes = 0

        try:
            frame_width, frame_height = video_capture.size
        except AttributeError:
            pass

        try:
            duration_seconds = video_capture.duration_seconds
        except AttributeError:
            pass

        try:
            size_bytes = path.getsize(full_path)
        except AttributeError:
            pass

        instance.starts_at = starts_at
        instance.finish_at = finish_at
        instance.frame_width = frame_width
        instance.frame_height=frame_height
        instance.size_bytes=size_bytes
        instance.duration_seconds = duration_seconds

        video_capture.release()


def create_video_thumbs(instance: VideoRecord):

    full_path = instance.full_path
    video_capture = VideoCapture(full_path)
    video_capture.open()
    if video_capture.is_opened():
        thumbs_frames = []
        try:
            thumbs_frames = video_capture.create_thumbs(
                count=settings.VIDEO_THUMBS_COUNT,
                size=settings.VIDEO_THUMBS_SIZE
            )
        except:
            print(f'Unable to create thumbs for video file "{full_path}".')

        for i, thumb in enumerate(thumbs_frames):
            thumb_filename = f'thumb_{instance.pk}_{i}.jpg'
            cv.imwrite(path.join(settings.MEDIA_ROOT, settings.VIDEO_THUMBS_PATH, thumb_filename), thumb)
            VideoThumb.objects.create(
                image=path.join(settings.VIDEO_THUMBS_PATH, thumb_filename),
                video=instance
            )

        video_capture.release()
