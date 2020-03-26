from rest_framework import serializers
from .abstracts import MaskFieldsSerializer
from ..models import Subject, SubjectSegment, Camera, VideoRecord, Task, Face


class SubjectSerializer(MaskFieldsSerializer):

    faces = serializers.PrimaryKeyRelatedField(
        queryset=Face.objects.all(),
        many=True,
        required=False
    )
    full_name = serializers.CharField(read_only=True)
    image = serializers.ImageField(read_only=True)
    age = serializers.IntegerField(read_only=True)
    last_name = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    birthdate = serializers.DateField(required=False, allow_null=True)
    timestamp = serializers.DateTimeField(read_only=True)
    sex = serializers.CharField(required=False, allow_blank=True)
    skin = serializers.CharField(required=False, allow_blank=True)
    pred_sex = serializers.CharField(read_only=True)
    pred_sex_score = serializers.FloatField(read_only=True)
    pred_age = serializers.IntegerField(read_only=True)
    pred_age_var = serializers.FloatField(read_only=True)

    class Meta:
        model = Subject
        fields = (
            'id',
            'faces',
            'name',
            'image',
            'last_name',
            'full_name',
            'age',
            'birthdate',
            'sex',
            'pred_sex',
            'pred_sex_score',
            'pred_age',
            'pred_age_var',
            'skin',
            'created_at',
            'updated_at',
            'timestamp'
        )
        read_only_fields = (
            'id',
            'created_at',
            'updated_at',
            'full_name',
            'age',
            'image',
            'pred_sex',
            'pred_sex_score',
            'pred_age',
            'pred_age_var',
            'timestamp'
        )


class SubjectSegmentSerializer(MaskFieldsSerializer):

    disk_cached = serializers.BooleanField(
        allow_null=True,
        required=False,
        help_text='Indicates if linked train data if stored in disk.'
    )

    title = serializers.CharField(
        help_text='The segment title.'
    )

    name = serializers.CharField(
        required=False,
        allow_blank=True
    )

    naming = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    min_age = serializers.IntegerField(required=False, allow_null=True, min_value=1, max_value=100)
    max_age = serializers.IntegerField(required=False, allow_null=True, min_value=1, max_value=100)
    min_timestamp = serializers.DateTimeField(required=False, allow_null=True)
    max_timestamp = serializers.DateTimeField(required=False, allow_null=True)
    sex = serializers.CharField(required=False, allow_blank=True)
    skin = serializers.CharField(required=False, allow_blank=True)
    count = serializers.IntegerField(read_only=True)

    cameras = serializers.PrimaryKeyRelatedField(
        queryset=Camera.objects.all(),
        many=True,
        required=False
    )

    videos = serializers.PrimaryKeyRelatedField(
        queryset=VideoRecord.objects.all(),
        many=True,
        required=False
    )

    tasks = serializers.PrimaryKeyRelatedField(
        queryset=Task.objects.all(),
        many=True,
        required=False
    )

    class Meta:
        model = SubjectSegment
        fields = (
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
        )
        read_only_fields = (
            'id',
            'count'
        )