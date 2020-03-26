from rest_framework import serializers

from ..models import Tag
from .abstracts import MaskFieldsSerializer


class TagSerializer(MaskFieldsSerializer):

    class Meta:
        model = Tag
        fields = (
            'id',
            'name',
            'model',
            'created_at',
            'updated_at',
        )
        read_only_fields = (
            'id',
            'created_at',
            'updated_at',
        )
