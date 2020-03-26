import shutil
from datetime import timedelta
from os import path, mkdir
from uuid import uuid4

import cv2 as cv
import numpy as np
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from faker import Faker

from ..models import (
    Face,
    Frame,
    Subject,
    SubjectSegment,
    VideoRecord,
    Camera,
    Notification,
    Stat,
    Recognition
)

TZ = timezone.get_current_timezone()

FAKER = Faker()

CURR_DIR = path.abspath(path.dirname(__file__))
FACE_IMAGE_PATH = path.join(CURR_DIR, 'data/face.jpg')
FRAME_IMAGE_PATH = path.join(CURR_DIR, 'data/frame.jpg')
VIDEO_LARGE_PATH = path.join(CURR_DIR, 'data/video_large.mp4')
VIDEO_SMALL_PATH = path.join(CURR_DIR, 'data/video_small.mp4')


def filter_keys(data: dict, keys: list):
    data_filtered = {}
    for key in keys:
        if key in data.keys():
            data_filtered[key] = data[key]
    return data_filtered


def create_video_file(file_path, media_path):
    _, ext = path.splitext(file_path)
    video_name = f'video_{uuid4()}{ext}'
    video_dir = path.join(settings.MEDIA_ROOT, media_path)
    if not path.exists(video_dir):
        mkdir(video_dir)
    shutil.copy2(file_path, path.join(video_dir, video_name))
    return video_name


def create_image_file(file_path, media_path):
    image_name = f'face_{uuid4()}.jpg'
    rel_path = path.join(media_path, image_name)
    full_path = path.join(settings.MEDIA_ROOT, rel_path)
    image = cv.imread(file_path)
    cv.imwrite(full_path, image)
    return rel_path


class ModelFactory:

    model_cls = None
    MODEL_REQUIRED_FIELDS = []
    API_REQUIRED_FIELDS = []
    API_READ_FIELDS = []

    def create_instance(self, full: bool = True):
        data = self.instance_data()
        if not full:
            data = filter_keys(data, self.MODEL_REQUIRED_FIELDS)
        return self.model_cls.objects.create(**data)

    def create_instances(self, full: bool = True, count: int = 5):
        return [self.create_instance(full) for _ in range(count)]

    def instance_data(self):
        return {}

    def api_post_data(self, full: bool = True):
        data = {}
        if not full:
            return filter_keys(data, self.API_REQUIRED_FIELDS)
        return data


class FrameFactory(ModelFactory):

    model_cls = Frame
    MODEL_REQUIRED_FIELDS = ['image']
    API_REQUIRED_FIELDS = ['image']
    API_READ_FIELDS = [
        'id',
        'image',
        'timestamp',
        'faces'
    ]

    def instance_data(self):
        return dict(
            image=create_image_file(
                FRAME_IMAGE_PATH,
                settings.FACES_IMAGES_PATH
            ),
            timestamp=timezone.now()
        )

    def api_post_data(self, full: bool = True):
        with open(FRAME_IMAGE_PATH, 'rb') as image_file:
            image = SimpleUploadedFile(
                'frame.jpg',
                image_file.read(),
                content_type="image/[jpg,png,gif]"
            )
            data = {
                'image': image,
                'timestamp': timezone.now()
            }

        if not full:
            return filter_keys(data, self.API_REQUIRED_FIELDS)

        return data


class FaceFactory(ModelFactory):

    model_cls = Face
    MODEL_REQUIRED_FIELDS = []
    API_REQUIRED_FIELDS = ['image']
    API_READ_FIELDS = [
        'id',
        'image',
        'frame',
        'box',
        'subject',
        'created_at',
        'timestamp'
    ]

    def instance_data(self):
        frame_factory = FrameFactory()
        subject_factory = SubjectFactory()
        return dict(
            frame=frame_factory.create_instance(),
            image=create_image_file(
                FACE_IMAGE_PATH,
                settings.FACES_IMAGES_PATH
            ),
            subject=subject_factory.create_instance(),
            box=(0, 0, 1, 1),
            embeddings=np.random.uniform(0, 1, 512),
            landmarks=np.random.uniform(0, 1, 10),
            timestamp=timezone.now()
        )

    def api_post_data(self, full: bool = True):
        subject_factory = SubjectFactory()
        with open(FACE_IMAGE_PATH, 'rb') as image_file:
            image = SimpleUploadedFile(
                'face.jpg',
                image_file.read(),
                content_type="image/[jpg,png,gif]"
            )
            data = {
                'image': image,
                'subject': subject_factory.create_instance().pk
            }

        if not full:
            return filter_keys(data, self.API_REQUIRED_FIELDS)

        return data


class CameraFactory(ModelFactory):

    model_cls = Camera
    MODEL_REQUIRED_FIELDS = ['name', 'stream_url']
    API_REQUIRED_FIELDS = ['name', 'stream_url']
    API_READ_FIELDS = [
        'id',
        'created_at',
        'updated_at',
        'stream_url',
        'name',
        'location_lat',
        'location_lon',
        'address',
        'running_tasks',
        'frames_count',
        'processing_time',
        'frame_rate',
        'faces_count',
        'last_task_at'
    ]

    def instance_data(self):
        return dict(
            stream_url=VIDEO_LARGE_PATH,
            name=f'Camera {FAKER.pyint()}',
            location_lat=FAKER.pydecimal(min_value=-90, max_value=90),
            location_lon=FAKER.pydecimal(min_value=-180, max_value=180),
            address=FAKER.address(),
        )

    def api_post_data(self, full: bool = True):
        data = self.instance_data()
        if not full:
            return filter_keys(data, self.API_REQUIRED_FIELDS)
        return data


class VideoFactory(ModelFactory):

    model_cls = VideoRecord
    MODEL_REQUIRED_FIELDS = ['path']
    API_REQUIRED_FIELDS = ['path']
    API_READ_FIELDS = [
        'id',
        'starts_at',
        'finish_at',
        'created_at',
        'updated_at',
        'frame_width',
        'frame_height',
        'duration_seconds',
        'size',
        'url',
        'thumbs',
        'running_tasks',
        'frames_count',
        'processing_time',
        'frame_rate',
        'faces_count',
        'last_task_at'
    ]

    def instance_data(self):
        return dict(
            path=create_video_file(
                VIDEO_SMALL_PATH,
                settings.VIDEO_RECORDS_PATH
            ),
            starts_at=timezone.now(),
            finish_at=timezone.now() + timedelta(seconds=60)
        )

    def api_post_data(self, full: bool = True):
        data = self.instance_data()
        if not full:
            return filter_keys(data, self.API_REQUIRED_FIELDS)
        return data


class SubjectFactory(ModelFactory):

    model_cls = Subject
    MODEL_REQUIRED_FIELDS = []
    API_REQUIRED_FIELDS = ['name']
    API_READ_FIELDS = [
        'id',
        'faces',
        'name',
        'last_name',
        'full_name',
        'age',
        'birthdate',
        'sex',
        'skin',
        'created_at',
        'updated_at',
        'task'
    ]

    def instance_data(self):
        return dict(
            name=FAKER.first_name(),
            last_name=FAKER.last_name(),
            birthdate=FAKER.date(),
            sex=Subject.SEX_MAN,
            skin=Subject.SKIN_WHITE,
        )

    def api_post_data(self, full: bool = True):
        data = self.instance_data()
        if not full:
            return filter_keys(data, self.API_REQUIRED_FIELDS)
        return data


class NotificationFactory(ModelFactory):

    model_cls = Notification
    MODEL_REQUIRED_FIELDS = [
        'category',
        'dtype',
        'title',
        'message',
        'resource',
        'seen'
    ]
    API_REQUIRED_FIELDS = []
    API_READ_FIELDS = [
        'id',
        'category',
        'dtype',
        'title',
        'message',
        'timestamp',
        'resource',
        'seen'
    ]

    def instance_data(self):
        return dict(
            category=Notification.CATEGORY_TASK,
            dtype=Notification.DTYPE_ERROR,
            title=FAKER.text(),
            message=FAKER.text(),
            resource=FAKER.pyint(),
            seen=FAKER.pybool(),
        )


class StatFactory(ModelFactory):

    model_cls = Stat
    MODEL_REQUIRED_FIELDS = [
        'category',
        'dtype',
        'title',
        'message',
        'resource',
        'seen'
    ]
    API_REQUIRED_FIELDS = []
    API_READ_FIELDS = [
        'id',
        'name',
        'timestamp',
        'updated_at',
        'value',
        'resolution'
    ]

    def instance_data(self):
        return dict(
            name=FAKER.text(),
            timestamp=FAKER.past_datetime(tzinfo=TZ),
            value=FAKER.pyfloat(),
            resolution=FAKER.random_element(Stat.RESOLUTION_CHOICES)[0]
        )


class SubjectSegmentFactory(ModelFactory):

    model_cls = SubjectSegment
    MODEL_REQUIRED_FIELDS = ['title']
    API_REQUIRED_FIELDS = ['title']
    API_READ_FIELDS = [
        'id',
        'disk_cached',
        'title',
        'name',
        'naming',
        'last_name',
        'min_age',
        'max_age',
        'min_timestamp',
        'max_timestamp',
        'sex',
        'skin',
        'count',
        'cameras',
        'videos',
        'tasks'
    ]

    def create_instance(self, full: bool = True):

        data = self.instance_data()
        videos = data.pop('videos')
        cameras = data.pop('cameras')
        if not full:
            data = filter_keys(data, self.MODEL_REQUIRED_FIELDS)
        instance = self.model_cls.objects.create(**data)
        instance.videos.add(*videos)
        instance.cameras.add(*cameras)

        return instance

    def instance_data(self):
        camera_factory = CameraFactory()
        video_factory = VideoFactory()
        return dict(
            disk_cached=True,
            title=f'Segment {FAKER.pyint()}',
            name=FAKER.name(),
            naming=SubjectSegment.NAMING_NAMED,
            last_name=FAKER.last_name(),
            min_birthdate=FAKER.past_date().isoformat(),
            max_birthdate=FAKER.future_date().isoformat(),
            min_timestamp=FAKER.past_datetime(tzinfo=TZ).isoformat(),
            max_timestamp=FAKER.future_datetime(tzinfo=TZ).isoformat(),
            sex=Subject.SEX_MAN,
            skin=Subject.SKIN_WHITE,
            count=0,
            model_path=f'Segment_{uuid4()}.npz',
            updated_at=timezone.now().isoformat(),
            cameras=[camera_factory.create_instance()],
            videos=[video_factory.create_instance()]
        )

    def api_post_data(self, full: bool = True):
        camera_factory = CameraFactory()
        video_factory = VideoFactory()
        data = dict(
            disk_cached=True,
            title=f'Segment {FAKER.pyint()}',
            name=FAKER.name(),
            naming=SubjectSegment.NAMING_NAMED,
            last_name=FAKER.last_name(),
            min_age=1,
            max_age=FAKER.pyint(min_value=1, max_value=70),
            min_timestamp=FAKER.past_datetime(tzinfo=TZ).isoformat(),
            max_timestamp=FAKER.future_datetime(tzinfo=TZ).isoformat(),
            sex=Subject.SEX_MAN,
            skin=Subject.SKIN_WHITE,
            cameras=[camera_factory.create_instance().pk],
            videos=[video_factory.create_instance().pk]
        )
        if not full:
            return filter_keys(data, self.API_REQUIRED_FIELDS)
        return data


class RecognitionFactory(ModelFactory):

    model_cls = Recognition
    MODEL_REQUIRED_FIELDS = ['face']
    API_REQUIRED_FIELDS = ['face']
    API_READ_FIELDS = [
        'id',
        'sim_thresh',
        'max_matches',
        'created_at',
        'face',
        'segments',
        'filter',
        'matches'
    ]

    def instance_data(self):
        face_factory = FaceFactory()
        return dict(
            sim_thresh=FAKER.pyfloat(min_value=0, max_value=1),
            max_matches=FAKER.pyint(min_value=1, max_value=10),
            face=face_factory.create_instance()
        )

    def api_post_data(self, full: bool = True):
        face_factory = FaceFactory()
        data = dict(
            sim_thresh=FAKER.pyfloat(min_value=0, max_value=1),
            max_matches=FAKER.pyint(min_value=1, max_value=10),
            face=face_factory.create_instance().pk
        )
        if not full:
            return filter_keys(data, self.API_REQUIRED_FIELDS)
        return data
