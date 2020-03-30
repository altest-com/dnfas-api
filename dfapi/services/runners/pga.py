from time import time, sleep
from typing import List

import cv2 as cv
from django.conf import settings
from dnfal.settings import Settings
from dnfal.vision import FacesVision
from dnfal.amazon.faces import analyze_faces, AwsFace

from .task import TaskRunner, PAUSE_DURATION, PROGRESS_UPDATE_INTERVAL
from ..subjects import pred_sexage
from ...models import (
    Subject,
    Face,
    PgaTaskConfig,
    Task
)

genderage_weights_path = settings.DNFAL_MODELS_PATHS['genderage_predictor']


class PgaTaskRunner(TaskRunner):

    def __init__(self, task: Task, daemon: bool = True):
        super().__init__(task, daemon)

        task = self.task

        se = Settings()

        se.force_cpu = settings.DNFAL_FORCE_CPU
        se.genderage_weights_path = genderage_weights_path
        se.face_align_size = 256

        self.task_config: PgaTaskConfig = PgaTaskConfig(**task.config)

        if self.task_config.method == PgaTaskConfig.METHOD_DNFAl:
            self.faces_vision: FacesVision = FacesVision(se)

        self._run: bool = False
        self._pause: bool = False

    def main_run(self):

        config = self.task_config

        if config.overwrite:
            faces_queryset = Face.objects.all()
        else:
            faces_queryset = (
                Face.objects.filter(pred_sex__exact='') |
                Face.objects.filter(pred_age__isnull=True)
            )

        faces_queryset = faces_queryset.exclude(image__isnull=True)

        if config.min_created_at is not None:
            faces_queryset = faces_queryset.filter(
                created_at__gt=config.min_created_at
            )

        if config.max_created_at is not None:
            faces_queryset = faces_queryset.filter(
                created_at__lt=config.max_created_at
            )

        batch_size = 36
        faces_count = 0
        started_at = time()
        total = faces_queryset.count()
        faces_batch = []

        self._run = True

        for face in faces_queryset.iterator():
            if not self._run:
                break

            while self._pause:
                sleep(PAUSE_DURATION)

            faces_batch.append(face)
            faces_count += 1
            last_face = faces_count == total

            if len(faces_batch) == batch_size or last_face:

                if config.method == PgaTaskConfig.METHOD_DNFAl:
                    self.dnfal_face_analysis(faces_batch)
                elif config.method == PgaTaskConfig.METHOD_AWS:
                    self.aws_face_analysis(faces_batch)

                faces_batch = []
                now = time()
                elapsed = now - self.last_progress_update
                if elapsed > PROGRESS_UPDATE_INTERVAL or last_face:
                    self.last_progress_update = now
                    self.task.progress = 100 * faces_count / total
                    info = self.task.info
                    info['faces_count'] = faces_count
                    info['processing_time'] = now - started_at
                    self.send_progress()

        # Update subjects
        subjects_queryset = Subject.objects.all()

        if self.task_config.min_created_at is not None:
            subjects_queryset = subjects_queryset.filter(
                faces_created_at__gt=self.task_config.min_created_at
            )
        if self.task_config.max_created_at is not None:
            subjects_queryset = subjects_queryset.filter(
                faces_created_at__lt=self.task_config.max_created_at
            )

        for subject in subjects_queryset.iterator():
            if not self._run:
                break
            while self._pause:
                sleep(PAUSE_DURATION)

            pred_age, age_var, pred_sex, sex_score = pred_sexage(subject)
            if pred_age is not None:
                subject.pred_age = int(pred_age)
                subject.pred_age_var = age_var

            if pred_sex:
                subject.pred_sex = pred_sex
                subject.pred_sex_score = sex_score

            if pred_age is not None or pred_sex:
                subject.save(update_fields=[
                    'pred_sex',
                    'pred_sex_score',
                    'pred_age',
                    'pred_age_var'
                ])

    def dnfal_face_analysis(self, faces: List[Face]):
        genderage_predictor = self.faces_vision.genderage_predictor
        face_aligner = self.faces_vision.face_aligner

        faces_images = []
        faces_inds = []
        for ind, face in enumerate(faces):
            face_image = face.image
            landmarks = face.landmarks
            if face_image is not None and len(landmarks):
                face_image = cv.imread(face_image.path)
                face_image_align, _ = face_aligner.align(face_image, landmarks)
                faces_images.append(face_image_align)
                faces_inds.append(ind)

                # color = (0, 255, 0)
                # for point in landmarks:
                #     point = (int(point[0]), int(point[1]))
                #     cv.circle(face_image, point, 2, color, -1)
                # cv.imshow('Face', face_image)
                # ret = cv.waitKey()
                # cv.imshow('Face aligned', face_image_align)
                # ret = cv.waitKey()

        n_images = len(faces_images)
        if n_images:
            (
                genders,
                genders_scores,
                ages,
                ages_vars
            ) = genderage_predictor.predict(faces_images)

            for ind in range(n_images):
                face = faces[faces_inds[ind]]
                if genders[ind] == genderage_predictor.GENDER_WOMAN:
                    face.pred_sex = Face.SEX_WOMAN
                elif genders[ind] == genderage_predictor.GENDER_MAN:
                    face.pred_sex = Face.SEX_MAN

                face.pred_sex_score = genders_scores[ind]
                face.pred_age = int(ages[ind])
                face.pred_age_var = ages_vars[ind]

                face.save(update_fields=[
                    'pred_sex',
                    'pred_sex_score',
                    'pred_age',
                    'pred_age_var'
                ])

    @staticmethod
    def aws_face_analysis(faces: List[Face]):

        faces_images = []
        faces_inds = []
        for ind, face in enumerate(faces):
            face_image = face.image
            if face_image is not None:
                face_image = cv.imread(face_image.path)
                faces_images.append(face_image)
                faces_inds.append(ind)

        n_images = len(faces_images)
        if n_images:
            aws_faces = analyze_faces(faces_images)

            for ind in range(n_images):
                aws_face = aws_faces[ind]
                if aws_face is not None:
                    face = faces[faces_inds[ind]]
                    if aws_face.gender == AwsFace.GENDER_WOMAN:
                        face.pred_sex = Face.SEX_WOMAN
                    elif aws_face.gender == AwsFace.GENDER_MAN:
                        face.pred_sex = Face.SEX_MAN

                    face.pred_sex_score = aws_face.gender_score

                    age_mean = 0.5 * (aws_face.age_high + aws_face.age_low)
                    age_var = 0.5 * (aws_face.age_high - aws_face.age_low)
                    face.pred_age = int(age_mean)
                    face.pred_age_var = int(age_var)

                    face.save(update_fields=[
                        'pred_sex',
                        'pred_sex_score',
                        'pred_age',
                        'pred_age_var'
                    ])

    def pause(self):
        self._pause = True
        super().pause()

    def resume(self):
        self._pause = False
        super().resume()

    def stop(self):
        self._run = False
        super().stop()

    def kill(self):
        self._run = False
        super().kill()

    def failed(self):
        self._run = False
        super().failed()