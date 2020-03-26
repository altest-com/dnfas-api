from .abstracts import MaskFieldsSerializer
from ..models import Stat


class StatSerializer(MaskFieldsSerializer):

    class Meta:
        model = Stat
        fields = '__all__'
        read_only_fields = (
            'id',
            'name',
            'timestamp',
            'updated_at',
            'value',
            'resolution'
        )