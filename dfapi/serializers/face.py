import base64
import binascii

import numpy as np
from dnfal.engine import EMBEDDINGS_LENGTH
from rest_framework import serializers

from .abstracts import MaskFieldsSerializer
from ..models import Face, Frame


class NpArrayField(serializers.Field):

    def __init__(self, **kwargs):
        self.dtype = kwargs.pop('dtype', np.float32)
        self.shape = kwargs.pop('shape', None)
        super().__init__(**kwargs)

    def to_representation(self, value):
        data = np.frombuffer(value, self.dtype)
        if self.shape is not None:
            data = data.reshape(self.shape)

        return data.tolist()

    def to_internal_value(self, data):
        if not isinstance(data, str):
            msg = f'Incorrect type. Expected a string, but got {type(data).__name__}.'
            raise serializers.ValidationError(msg)

        try:
            data_bytes = base64.b64decode(data, validate=True)
        except binascii.Error:
            msg = f'Incorrect type. Can not decode the data to base64.'
            raise serializers.ValidationError(msg)

        try:
            self.to_representation(data_bytes)
        except ValueError:
            msg = f'Incorrect type. Can not convert the data to numpy array.'
            raise serializers.ValidationError(msg)

        return data_bytes


# class FaceFrameSerializer(serializers.ModelSerializer):
#
#     image = serializers.ImageField()
#     timestamp = serializers.DateTimeField(required=False, allow_null=True)
#
#     class Meta:
#         model = Frame
#         fields = ('id', 'image', 'timestamp')
#         read_only_fields = ('id',)


class FaceSerializer(MaskFieldsSerializer):

    image = serializers.ImageField(required=True, allow_null=True)
    frame = serializers.PrimaryKeyRelatedField(
        required=False,
        allow_null=True,
        queryset=Frame.objects.all()
    )

    box = serializers.ListSerializer(
        required=False,
        allow_null=True,
        child=serializers.FloatField(),
        read_only=True
    )

    pred_sex = serializers.ChoiceField(read_only=True, choices=Face.SEX_CHOICES)
    pred_age = serializers.IntegerField(read_only=True)

    class Meta:
        model = Face
        fields = (
            'id',
            'image',
            'frame',
            'box',
            'subject',
            'created_at',
            'pred_sex',
            'pred_age',
            'timestamp',
            'task'
        )
        read_only_fields = (
            'id',
            'created_at',
            'box',
            'pred_sex',
            'pred_age',
            'task'
        )


# class FrameFaceSerializer(serializers.ModelSerializer):
#
#     class Meta:
#         model = Face
#         fields = ('id', 'frame', 'image', 'box', 'subject')
#         read_only_fields = ('id', 'frame', 'image', 'box', 'subject')


class FrameSerializer(MaskFieldsSerializer):

    image = serializers.ImageField()
    timestamp = serializers.DateTimeField(required=False, allow_null=True)
    faces = serializers.PrimaryKeyRelatedField(
        many=True,
        read_only=True
    )

    class Meta:
        model = Frame
        fields = ('id', 'image', 'timestamp', 'faces')
        read_only_fields = ('id', 'faces')
