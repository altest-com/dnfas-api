from datetime import datetime, timedelta
from typing import List
from time import time

import numpy as np
from django.utils.timezone import make_aware
from dnfal.engine import similarity_to_distance
from dnfal.clustering import hcg_cluster

from .task import TaskRunner
from ...models import (
    Subject,
    Face,
    FclTaskConfig,
    Task
)


class FclTaskRunner(TaskRunner):

    def __init__(self, task: Task, daemon: bool = True):
        super().__init__(task, daemon)

        self.task_config: FclTaskConfig = FclTaskConfig(**self.task.config)
        self._run: bool = False
        self._pause: bool = False

    def main_run(self):

        started_at = time()

        config = self.task_config

        faces_queryset = Face.objects.exclude(
            embeddings_bytes__isnull=True
        )

        back_time = {}

        if config.filter_back_weeks:
            back_time['weeks'] = config.filter_back_weeks

        if config.filter_back_days:
            back_time['days'] = config.filter_back_days

        if config.filter_back_hours:
            back_time['hours'] = config.filter_back_hours

        if len(back_time):
            now = make_aware(datetime.now())
            min_created_at = now - timedelta(**back_time)
            faces_queryset = faces_queryset.filter(
                created_at__gt=min_created_at
            )

        if config.filter_min_date:
            faces_queryset = faces_queryset.filter(
                created_at__date__gt=config.filter_min_date
            )

        if config.filter_max_date:
            faces_queryset = faces_queryset.filter(
                created_at__date__lt=config.filter_max_date
            )

        if config.filter_min_time:
            faces_queryset = faces_queryset.filter(
                created_at__time__gt=config.filter_min_time
            )

        if config.filter_max_time:
            faces_queryset = faces_queryset.filter(
                created_at__time__lt=config.filter_max_time
            )

        if len(config.filter_tasks) and len(config.filter_tasks_tags):
            faces_queryset = faces_queryset.filter(
                task__in=config.filter_tasks
            ) | faces_queryset.filter(
                task__tags__in=config.filter_tasks_tags
            )
        else:
            if len(config.filter_tasks):
                faces_queryset = faces_queryset.filter(
                    task__in=config.filter_tasks
                )
            if len(config.filter_tasks_tags):
                faces_queryset = faces_queryset.filter(
                    task__tags__in=config.filter_tasks_tags
                )

        faces_queryset = faces_queryset.order_by('created_at')

        top_dist_thr = similarity_to_distance(self.task_config.top_dist_thr)
        low_dist_thr = similarity_to_distance(self.task_config.low_dist_thr)
        edge_thr = self.task_config.edge_thr
        linkage = self.task_config.linkage
        timestamp_thr = self.task_config.memory_seconds

        embeddings = []
        timestamps = [] if timestamp_thr else None
        queryset_pks = []
        for face in faces_queryset.iterator():
            embeddings.append(face.embeddings)
            queryset_pks.append(face.id)
            if timestamps is not None:
                timestamps.append(face.created_at.timestamp())

        embeddings = np.array(embeddings, np.float32)

        if timestamps is not None:
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)
            if max_timestamp - min_timestamp < timestamp_thr:
                timestamps = None
            else:
                timestamps = np.array(timestamps).reshape((-1, 1))

        clusters = hcg_cluster(
            features=embeddings,
            timestamps=timestamps,
            distance_thr=(top_dist_thr, low_dist_thr),
            timestamp_thr=timestamp_thr,
            edge_thr=edge_thr,
            linkage=linkage
        )

        queryset_pks = set(queryset_pks)

        for cluster in clusters:
            faces_cluster = []
            cluster_pks = set()
            for face_ind in cluster:
                face_ind = int(face_ind)
                face = faces_queryset[face_ind]
                faces_cluster.append(face)
                cluster_pks.update([face.id])
                if face.subject is not None:
                    for face in face.subject.faces.all():
                        if (
                            face.id not in queryset_pks and
                            face.id not in cluster_pks
                        ):
                            faces_cluster.append(face)
                            cluster_pks.update([face.id])

            self.merge_faces(faces_cluster)

        processing_time = time() - started_at
        faces_count = faces_queryset.count()

        self.task.info['processing_time'] = processing_time
        self.task.info['faces_count'] = faces_count

    @staticmethod
    def merge_faces(faces_cluster: List[Face]):
        subject_data = {
            'name': '',
            'last_name': '',
            'birthdate': None,
            'sex': '',
            'skin': ''
        }
        for face in faces_cluster:
            subject = face.subject
            if subject is not None:
                subject.faces.clear()
                for key in subject_data:
                    value = getattr(subject, key)
                    if value:
                        subject_data[key] = value

                subject.delete()

        subject = Subject.objects.create(**subject_data)
        subject.faces.set(faces_cluster)

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