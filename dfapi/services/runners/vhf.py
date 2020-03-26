from dnfal import mtypes
from dnfal.engine import VideoAnalyzer
from dnfal.settings import Settings

from .task import logger
from .vdf import VdfTaskRunner, create_face
from ...models import (
    Subject,
    Face,
    HuntMatch,
    VhfTaskConfig,
    Task
)


def update_face_hunt(face: mtypes.Face, task_id: int):

    if face.subject is None:
        logger.error('Invalid operation. Face subject can not be empty.')
        return

    hunt_match_id = face.subject.data.get('hunt_key', None)
    try:
        hunt_match: HuntMatch = HuntMatch.objects.get(pk=hunt_match_id)
    except HuntMatch.DoesNotExist:
        logger.error(f'Hunt match <{hunt_match_id}> does not exist.')
        return
    subject_instance = hunt_match.matched_subject
    if subject_instance is None:
        subject_instance = Subject.objects.create()
        face.subject.data['subject_id'] = subject_instance.pk
        hunt_match.matched_subject = subject_instance
        hunt_match.save(update_fields=['matched_subject'])
    create_face(face, subject_instance.pk, task_id)


class VhfTaskRunner(VdfTaskRunner):

    def __init__(self, task: Task, daemon: bool = True):
        super().__init__(task, daemon)

    def init_vision(self, vision_settings: Settings):
        task_config = VhfTaskConfig(**self.task.config)

        embeddings = []
        keys = []
        subjects = Subject.objects.filter(
            pk__in=task_config.hunted_subjects
        )

        for subject in subjects:
            hunt_match = HuntMatch.objects.create(
                target_subject=subject,
                task=self.task
            )
            for face in subject.faces.all():
                keys.append(hunt_match.pk)
                embeddings.append(face.embeddings)

        vision_settings.video_mode = VideoAnalyzer.MODE_HUNT
        vision_settings.video_hunt_embeddings = embeddings
        vision_settings.video_hunt_keys = keys

        super().init_vision(vision_settings)

    def on_subject_updated(self, face: Face):
        self.executor.submit(
            update_face_hunt,
            face=face,
            task_id=self.task.pk
        )