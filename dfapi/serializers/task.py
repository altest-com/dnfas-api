from rest_framework import serializers

from .abstracts import MaskFieldsSerializer
from ..models import (
    VideoRecord,
    Camera,
    Subject,
    Task,
    Tag,
    VTaskConfig,
    FclTaskConfig
)


class VTaskConfigSerializer(serializers.Serializer):

    video_source_type = serializers.ChoiceField(
        choices=VTaskConfig.VIDEO_SOURCE_CHOICES
    )
    video_source_id = serializers.IntegerField()
    start_at = serializers.FloatField(required=False)
    stop_at = serializers.FloatField(required=False)

    def validate(self, data):
        source_type = data['video_source_type']
        source_id = data['video_source_id']
        if (
            source_type == VTaskConfig.VIDEO_SOURCE_CAMERA and
            not Camera.objects.filter(pk=source_id)
        ) or (
            source_type == VTaskConfig.VIDEO_SOURCE_RECORD and
            not VideoRecord.objects.filter(pk=source_id)
        ):
            raise serializers.ValidationError(
                f'Video source <{source_type}> does not exists'
            )
        return super().validate(data)


class VdfTaskConfigSerializer(VTaskConfigSerializer):

    detection_min_height = serializers.IntegerField(required=False)
    detection_min_score = serializers.FloatField(required=False)
    similarity_thresh = serializers.FloatField(
        required=False,
        min_value=0,
        max_value=0.99999
    )
    max_frame_size = serializers.IntegerField(required=False)
    frontal_faces = serializers.BooleanField(required=False)
    video_detect_interval = serializers.FloatField(required=False)
    faces_time_memory = serializers.FloatField(required=False)
    store_face_frames = serializers.BooleanField(required=False)


class VhfTaskConfigSerializer(VdfTaskConfigSerializer):

    hunted_subjects = serializers.ListSerializer(
        child=serializers.IntegerField(),
        required=False
    )

    def validate_hunted_subjects(self, value):
        if len(value):
            subjects = Subject.objects.filter(pk__in=value)
            if len(subjects) != len(value):
                raise serializers.ValidationError(
                    f'Invalid hunted subjects IDs'
                )
        return value


class PgaTaskConfigSerializer(serializers.Serializer):

    min_created_at = serializers.DateTimeField(required=False, allow_null=True)
    max_created_at = serializers.DateTimeField(required=False, allow_null=True)
    overwrite = serializers.BooleanField(required=False)


class FclTaskConfigSerializer(serializers.Serializer):

    filter_back_weeks = serializers.IntegerField(
        required=False,
        allow_null=True,
        min_value=0
    )
    filter_back_days = serializers.IntegerField(
        required=False,
        allow_null=True,
        min_value=0
    )
    filter_back_minutes = serializers.IntegerField(
        required=False,
        allow_null=True,
        min_value=0
    )
    filter_min_date = serializers.DateField(
        required=False,
        allow_null=True
    )
    filter_max_date = serializers.DateField(
        required=False,
        allow_null=True
    )
    filter_min_time = serializers.TimeField(
        required=False,
        allow_null=True
    )
    filter_max_time = serializers.TimeField(
        required=False,
        allow_null=True
    )
    filter_tasks = serializers.ListSerializer(
        child=serializers.IntegerField(min_value=0),
        required=False
    )
    filter_tasks_tags = serializers.ListSerializer(
        child=serializers.IntegerField(min_value=0),
        required=False
    )
    top_dist_thr = serializers.FloatField(
        required=False,
        min_value=0,
        max_value=0.99999
    )
    low_dist_thr = serializers.FloatField(
        required=False,
        min_value=0,
        max_value=0.99999
    )
    edge_thr = serializers.FloatField(
        required=False,
        min_value=0,
        max_value=1
    )
    linkage = serializers.ChoiceField(
        required=False,
        choices=FclTaskConfig.LINKAGE_CHOICES
    )
    memory_seconds = serializers.IntegerField(
        required=False,
        allow_null=True,
        min_value=0
    )

    def validate(self, data):
        filter_tasks = data['filter_tasks']
        filter_tasks_tags = data['filter_tasks_tags']

        for task_pk in filter_tasks:
            Task.objects.get(pk=task_pk)

        for tag_pk in filter_tasks_tags:
            Tag.objects.get(pk=tag_pk)

        return super().validate(data)


_config_serializers = {
    Task.TYPE_VIDEO_DETECT_FACES: VdfTaskConfigSerializer,
    Task.TYPE_VIDEO_HUNT_FACES: VhfTaskConfigSerializer,
    Task.TYPE_PREDICT_GENDERAGE: PgaTaskConfigSerializer,
    Task.TYPE_FACE_CLUSTERING: FclTaskConfigSerializer
}


class TaskSerializer(MaskFieldsSerializer):

    # Required
    name = serializers.CharField(max_length=255)
    task_type = serializers.ChoiceField(choices=Task.TYPE_CHOICES)

    # Optional
    schedule_start_at = serializers.DateTimeField(allow_null=True, required=False)
    schedule_stop_at = serializers.DateTimeField(allow_null=True, required=False)
    # repeat = serializers.BooleanField(required=False)
    repeat_days = serializers.CharField(max_length=7, required=False, allow_blank=True)
    config = serializers.JSONField(required=False, default=dict)
    tags = serializers.PrimaryKeyRelatedField(
        queryset=Tag.objects.all(),
        required=False,
        many=True
    )

    # Read only
    status = serializers.CharField(read_only=True)
    started_at = serializers.DateTimeField(read_only=True)
    finished_at = serializers.DateTimeField(read_only=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    progress = serializers.FloatField(read_only=True)
    worker = serializers.PrimaryKeyRelatedField(
        read_only=True
    )
    info = serializers.JSONField(read_only=True)

    def validate(self, data: dict):
        task_type = data['task_type']
        config = data.get('config', {})
        serializer_class = _config_serializers[task_type]
        serializer = serializer_class(data=config)
        serializer.is_valid(raise_exception=True)
        return data

    class Meta:
        model = Task
        fields = (
            'id',
            'name',
            'tags',
            'task_type',
            'schedule_start_at',
            'schedule_stop_at',
            'repeat_days',
            'config',
            'status',
            'started_at',
            'finished_at',
            'created_at',
            'updated_at',
            'progress',
            'worker',
            'info'
        )



