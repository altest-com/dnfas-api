import logging
import signal
import uuid
from multiprocessing import Process, Queue
from os import path
from queue import Empty as QueueEmptyError
from queue import Full as QueueFullError
from typing import List

import cv2 as cv
import numpy as np
from django import db
from django.conf import settings
from django.db.models import QuerySet
from dnfal.alignment import FaceAligner
from dnfal.genderage import GenderAgePredictor
from dnfal.settings import Settings
from dnfal.vision import FacesVision
from openpyxl import Workbook

from .exceptions import ServiceError
from ..models import (
    Face,
    Frame,
    Recognition,
    RecognitionMatch,
    SubjectSegment
)

logger_name = settings.LOGGER_NAME
logger = logging.getLogger(logger_name)

ENGINE_WAIT_TIMEOUT = 300


def predict_genderage(
    faces_id: List[int],
    genderage_predictor: GenderAgePredictor,
    face_aligner: FaceAligner
):

    faces = Face.objects.filter(pk__in=faces_id)
    faces_images = []
    faces_inds = []
    for ind, face in faces:
        face_image = face.image
        landmarks = face.landmarks
        if face_image is not None and len(landmarks):
            face_image = cv.imread(face_image.path)
            face_image_align, _ = face_aligner.align(face_image, landmarks)
            faces_images.append(face_image_align)
            faces_inds.append(ind)

    n_images = len(faces_images)
    if n_images:
        genders, _, ages, _ = genderage_predictor.predict(faces_images)
        for ind in range(n_images):
            face = faces[faces_inds[ind]]
            if genders[ind] == GenderAgePredictor.GENDER_WOMAN:
                face.pred_sex = Face.SEX_WOMAN
            elif genders[ind] == GenderAgePredictor.GENDER_MAN:
                face.pred_sex = Face.SEX_MAN

            face.pred_age = int(ages[ind])

            face.save(update_fields=['pred_sex', 'pred_age'])


def analyze_face(face_id: int, faces_vision: FacesVision):

    face = Face.objects.get(pk=face_id)
    face_image = cv.imread(face.image.path)

    if face_image is None:
        return face

    detected_faces, _ = faces_vision.frame_analyzer.find_faces(face_image)

    if len(detected_faces) != 1:
        return face

    face.box = detected_faces[0].box
    face.embeddings = detected_faces[0].embeddings
    face.landmarks = detected_faces[0].landmarks

    face.save()

    return face


def analyze_frame(frame_id: int, faces_vision: FacesVision):
    frame = Frame.objects.get(pk=frame_id)
    frame_image = cv.imread(frame.image.path)
    faces, _ = faces_vision.frame_analyzer.find_faces(frame_image)

    if len(faces) == 0:
        return []

    face_objects = []
    for face in faces:
        image_name = f'face_{uuid.uuid4()}.jpg'
        rel_path = path.join(settings.FACES_IMAGES_PATH, image_name)
        full_path = path.join(settings.MEDIA_ROOT, rel_path)
        cv.imwrite(full_path, face.image)

        face_objects.append(Face.objects.create(
            frame=frame,
            image=rel_path,
            box=face.box,
            embeddings=face.embeddings,
            landmarks=face.landmarks,
        ))


def recognize_face(
    recognition_id: int,
    faces_vision: FacesVision
):
    recognition: Recognition = Recognition.objects.get(pk=recognition_id)

    face_embeddings = recognition.face.embeddings.reshape((1, -1))

    subjects_embeddings = []
    subjects = []

    segments = recognition.segments.all()

    # if len(segments) == 0 and recognition.filter is not None:
    #     segments = [recognition.filter]
    # elif len(segments) == 0:
    #     segment, _ = SubjectSegment.objects.get_or_create(
    #         title=settings.DEFAULT_SEGMENT_TITLE,
    #         disk_cached=True
    #     )
    #     segments = [segment]

    if len(segments) == 0:
        segment, _ = SubjectSegment.objects.get_or_create(
            title=settings.DEFAULT_SEGMENT_TITLE,
            disk_cached=True
        )
        segments = [segment]

    for segment in segments:
        segment_embeddings, segment_subjects = segment.get_data()
        subjects_embeddings.append(segment_embeddings)
        subjects.append(segment_subjects)

    if len(segments) > 1:
        subjects_embeddings = np.vstack(subjects_embeddings)
        subjects = np.hstack(subjects)
    else:
        subjects_embeddings = subjects_embeddings[0]
        subjects = subjects[0]

    if len(subjects) == 0:
        return [], []

    faces_vision.face_matcher.similarity_threshold = float(recognition.sim_thresh)
    subject_ids, scores = faces_vision.face_matcher.match(
        x_test=face_embeddings,
        x_train=subjects_embeddings,
        y_train=subjects
    )

    subject_ids = subject_ids[0]
    scores = scores[0]

    if len(subject_ids) == 0:
        return [], []

    max_matches = recognition.max_matches
    if 0 < max_matches < len(subject_ids):
        subject_ids = subject_ids[0:max_matches]
        scores = scores[0:max_matches]

    for subject_id, score in zip(subject_ids, scores):
        RecognitionMatch.objects.create(
            subject_id=subject_id,
            score=score,
            recognition_id=recognition.pk
        )


class FaceAnalyzer:

    MAX_QUEUE_SIZE = 1000
    REQUEST_TIMEOUT = 30

    TASK_ANALYZE_FACE = 'analyze_face'
    TASK_ANALYZE_FRAME = 'analyze_frame'
    TASK_RECOGNIZE_FACE = 'recognize_face'
    TASK_PREDICT_GENDERAGE = 'predict_genderage'
    TASK_TERMINATE = 'terminate'

    def __init__(self):
        self.send_queue = Queue(self.MAX_QUEUE_SIZE)
        self.recv_queue = Queue(self.MAX_QUEUE_SIZE)
        self.process = None

        self._task_count = 0

    def start_process(self):
        if self.process is None or not self.process.is_alive():
            db.connections.close_all()
            self.process = Process(
                target=execute_task,
                kwargs={
                    'send_queue': self.recv_queue,
                    'recv_queue': self.send_queue,
                },
                daemon=True
            )
            self.process.start()

    def send_task(self, task_data: dict, timeout=None):
        self.start_process()
        try:
            self.send_queue.put(task_data, timeout=timeout)
            response: dict = self.recv_queue.get(
                timeout=self.REQUEST_TIMEOUT
            )
            if response['error'] is not None:
                raise ServiceError(response['error'])
            if response['task_id'] != task_data['task_id']:
                raise ServiceError('Invalid task id in response.')
        except QueueFullError:
            raise ServiceError('Task can no be completed. Task queue is full.')
        except QueueEmptyError:
            raise ServiceError('Task result could not be retrieved. Timeout error.')

    def analyze_face(self, face_id: int):
        self._task_count += 1
        task_data = {
            'task_name': self.TASK_ANALYZE_FACE,
            'task_id': self._task_count,
            'kwargs': {
                'face_id': face_id
            }
        }
        self.send_task(task_data, timeout=self.REQUEST_TIMEOUT)

    def predict_genderage(self, faces_id: List[int]):
        self._task_count += 1
        task_data = {
            'task_name': self.TASK_PREDICT_GENDERAGE,
            'task_id': self._task_count,
            'kwargs': {
                'faces_id': faces_id
            }
        }
        self.send_task(task_data, timeout=self.REQUEST_TIMEOUT)

    def analyze_frame(self, frame_id: int):
        self._task_count += 1
        task_data = {
            'task_name': self.TASK_ANALYZE_FRAME,
            'task_id': self._task_count,
            'kwargs': {
                'frame_id': frame_id
            }
        }
        self.send_task(task_data, timeout=self.REQUEST_TIMEOUT)

    def recognize_face(self, recognition_id: int):
        self._task_count += 1
        task_data = {
            'task_name': self.TASK_RECOGNIZE_FACE,
            'task_id': self._task_count,
            'kwargs': {
                'recognition_id': recognition_id
            }
        }
        self.send_task(task_data, timeout=self.REQUEST_TIMEOUT)

    def terminate(self):
        if self.process is not None and self.process.is_alive():
            self._task_count += 1
            task_data = {
                'task_name': self.TASK_TERMINATE,
                'task_id': self._task_count,
                'kwargs': None
            }
            self.send_task(task_data)


def execute_task(send_queue: Queue, recv_queue: Queue):

    def _handle_signal(_signal_number, _stack_frame):
        task_data = {
            'task_name': FaceAnalyzer.TASK_TERMINATE,
            'task_id': None,
            'kwargs': None
        }
        recv_queue.put(task_data)

    for signal_key in (signal.SIGQUIT, signal.SIGTERM, signal.SIGINT):
        signal.signal(signal_key, _handle_signal)

    se = Settings()

    se.force_cpu = settings.DNFAL_FORCE_CPU
    se.detector_weights_path = settings.DNFAL_MODELS_PATHS['face_detector']
    se.marker_weights_path = settings.DNFAL_MODELS_PATHS['face_marker']
    se.encoder_weights_path = settings.DNFAL_MODELS_PATHS['face_encoder']
    se.align_max_deviation = None
    se.detection_min_scores = 0.9
    se.marking_min_score = 0
    se.detection_min_size = 32

    genderage_weights_path = settings.DNFAL_MODELS_PATHS['genderage_predictor']

    genderage_predictor = None
    face_aligner = None

    if settings.DEBUG:
        se.log_to_console = True

    se.video_capture_source = None

    faces_vision = FacesVision(se)

    while True:
        try:
            task: dict = recv_queue.get(timeout=ENGINE_WAIT_TIMEOUT)
        except QueueEmptyError:
            break

        task_name = task['task_name']
        task_id = task['task_id']
        kwargs = task['kwargs']

        response_data = {
            'task_id': task_id,
            'error': None
        }

        if task_name == FaceAnalyzer.TASK_ANALYZE_FACE:
            face_id = kwargs['face_id']
            analyze_face(face_id=face_id, faces_vision=faces_vision)
        elif task_name == FaceAnalyzer.TASK_PREDICT_GENDERAGE:
            if genderage_predictor is None:
                genderage_predictor = GenderAgePredictor(genderage_weights_path)
            if face_aligner is None:
                face_aligner = FaceAligner(out_size=256)

            faces_id = kwargs['faces_id']
            predict_genderage(
                faces_id=faces_id,
                genderage_predictor=genderage_predictor,
                face_aligner=face_aligner
            )
        elif task_name == FaceAnalyzer.TASK_ANALYZE_FRAME:
            frame_id = kwargs['frame_id']
            analyze_frame(frame_id=frame_id, faces_vision=faces_vision)
        elif task_name == FaceAnalyzer.TASK_RECOGNIZE_FACE:
            recognition_id = kwargs['recognition_id']
            recognize_face(recognition_id=recognition_id, faces_vision=faces_vision)
        elif task_name == FaceAnalyzer.TASK_TERMINATE:
            break
        else:
            response_data['error'] = f'Invalid task name {task_name}'

        try:
            send_queue.put(response_data)
        except QueueFullError:
            pass


face_analyzer = FaceAnalyzer()

