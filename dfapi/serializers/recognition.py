from rest_framework import serializers
from .abstracts import MaskFieldsSerializer
from ..models import Recognition, RecognitionMatch, SubjectSegment
# from .subject import SubjectSegmentSerializer, SubjectSerializer


class RecognitionMatchSerializer(serializers.ModelSerializer):

    class Meta:
        model = RecognitionMatch
        fields = (
            'id',
            'score',
            'subject',
        )
        read_only_fields = (
            'id',
            'score',
            'subject'
        )


class RecognitionSerializer(MaskFieldsSerializer):

    matches = RecognitionMatchSerializer(
        many=True,
        read_only=True
    )

    segments = serializers.PrimaryKeyRelatedField(
        queryset=SubjectSegment.objects.all(),
        many=True,
        required=False
    )

    # filter = SubjectSegmentSerializer(
    #     allow_null=True,
    #     required=False
    # )

    class Meta:
        model = Recognition
        fields = (
            'id',
            'sim_thresh',
            'max_matches',
            'created_at',
            'face',
            'segments',
            'matches'
        )
        read_only_fields = (
            'id',
            'created_at',
            'matches'
        )

    # def create(self, validated_data):
    #     try:
    #         filter_data = validated_data.pop('filter')
    #     except KeyError:
    #         pass
    #     else:
    #         if len(filter_data):
    #             segment_serializer = SubjectSegmentSerializer(
    #                 data=filter_data,
    #                 context=self.context
    #             )
    #             segment_serializer.is_valid(raise_exception=True)
    #             segment = segment_serializer.save()
    #             validated_data['filter_id'] = segment.pk
    #
    #     return super().create(validated_data)
