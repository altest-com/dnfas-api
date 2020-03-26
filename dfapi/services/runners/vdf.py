import datetime
from datetime import datetime
from os import path
from time import time
from uuid import uuid4

from django.conf import settings
from django.utils.timezone import make_aware
from dnfal import mtypes
from dnfal.engine import VideoAnalyzer
from dnfal.settings import Settings
from dnfal.vision import FacesVision

from .task import TaskRunner, logger, PROGRESS_UPDATE_INTERVAL
from ...models import (
    Subject,
    Face,
    Frame,
    VideoRecord,
    Camera,
    VdfTaskConfig,
    Task
)


def create_frame(frame: mtypes.Frame):

    frame_name = f'frame_{uuid4()}.jpg'
    rel_path = path.join(settings.FACES_IMAGES_PATH, frame_name)
    full_path = path.join(settings.MEDIA_ROOT, rel_path)
    with open(full_path, 'wb') as f:
        f.write(frame.image_bytes)

    instance = Frame.objects.create(
        image=rel_path
    )

    frame.data['frame_id'] = instance.pk


def create_face(face: mtypes.Face, subject_id: int, task_id: int):

    frame_id = None
    if face.frame is not None:
        frame_id = face.frame.data.get('frame_id', None)
        if frame_id is None:
            create_frame(face.frame)
            frame_id = face.frame.data['frame_id']

    image_name = f'face_{uuid4()}.jpg'
    rel_path = path.join(settings.FACES_IMAGES_PATH, image_name)
    full_path = path.join(settings.MEDIA_ROOT, rel_path)
    with open(full_path, 'wb') as f:
        f.write(face.image_bytes)

    timestamp = make_aware(
        datetime.fromtimestamp(face.timestamp)
    )

    instance = Face.objects.create(
        subject_id=subject_id,
        task_id=task_id,
        frame_id=frame_id,
        image=rel_path,
        box=face.box,
        embeddings=face.embeddings,
        landmarks=face.landmarks,
        timestamp=timestamp
    )

    face.data['key'] = instance.pk


def update_face_detect(face: mtypes.Face, task_id: int):

    if face.subject is None:
        logger.error('Invalid operation. Face subject can not be empty.')
        return

    subject_id = face.subject.data.get('subject_id', None)
    if subject_id is None:
        subject_instance = Subject.objects.create()
        face.subject.data['subject_id'] = subject_instance.pk
    create_face(face, face.subject.data['subject_id'], task_id)


class VdfTaskRunner(TaskRunner):

    def __init__(self, task: Task, daemon: bool = True):
        super().__init__(task, daemon)

        task = self.task

        se = Settings()

        se.force_cpu = settings.DNFAL_FORCE_CPU
        se.detector_weights_path = settings.DNFAL_MODELS_PATHS['face_detector']
        se.marker_weights_path = settings.DNFAL_MODELS_PATHS['face_marker']
        se.encoder_weights_path = settings.DNFAL_MODELS_PATHS['face_encoder']

        task_config = VdfTaskConfig(**task.config)
        video_source_type = task_config.video_source_type

        if video_source_type == VdfTaskConfig.VIDEO_SOURCE_RECORD:
            video = VideoRecord.objects.get(pk=task_config.video_source_id)
            se.video_capture_source = video.full_path
            se.video_real_time = False
        elif video_source_type == VdfTaskConfig.VIDEO_SOURCE_CAMERA:
            camera = Camera.objects.get(pk=task_config.video_source_id)
            se.video_capture_source = camera.stream_url
            se.video_real_time = True

        se.video_mode = VideoAnalyzer.MODE_ALL
        se.video_start_at = task_config.start_at
        se.video_stop_at = task_config.stop_at
        se.detection_min_height = task_config.detection_min_height
        se.detection_min_score = task_config.detection_min_score
        se.similarity_thresh = task_config.similarity_thresh
        se.max_frame_size = task_config.max_frame_size
        se.video_detect_interval = task_config.video_detect_interval
        se.faces_time_memory = task_config.faces_time_memory
        se.store_face_frames = task_config.store_face_frames

        if task_config.frontal_faces:
            se.align_max_deviation = (0.4, 0.3)

        # noinspection PyTypeChecker
        self.faces_vision: FacesVision = None

        self.init_vision(se)

    def init_vision(self, vision_settings):
        self.faces_vision = FacesVision(vision_settings)

    def main_run(self):
        self.faces_vision.video_analyzer.run(
            frame_callback=self.on_frame,
            update_subject_callback=self.on_subject_updated
        )

    def on_subject_updated(self, face: Face):
        self.executor.submit(
            update_face_detect,
            face=face,
            task_id=self.task.pk
        )

    def on_frame(self):
        now = time()
        if (now - self.last_progress_update) > PROGRESS_UPDATE_INTERVAL:
            self.last_progress_update = now
            video_analyzer = self.faces_vision.video_analyzer
            info = self.task.info
            try:
                info['frames_count'] = video_analyzer.frames_count
                info['processing_time'] = video_analyzer.processing_time
                info['faces_count'] = video_analyzer.faces_count
                if video_analyzer.start_at > 0:
                    self.task.frame_rate = video_analyzer.frames_count / (
                        now - video_analyzer.start_at
                    )
            except Exception as err:
                logger.error(err)

            self.send_progress()

    def pause(self):
        self.faces_vision.video_analyzer.pause()
        super().pause()

    def resume(self):
        self.faces_vision.video_analyzer.pause()
        super().resume()

    def stop(self):
        self.faces_vision.video_analyzer.stop()
        super().stop()

    def kill(self):
        self.faces_vision.video_analyzer.stop()
        super().kill()

    def failed(self):
        self.faces_vision.video_analyzer.stop()
        super().failed()