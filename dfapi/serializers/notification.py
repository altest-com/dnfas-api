from .abstracts import MaskFieldsSerializer
from ..models import Notification


class NotificationSerializer(MaskFieldsSerializer):

    class Meta:
        model = Notification
        fields = '__all__'
        read_only_fields = (
            'id',
            'category',
            'dtype',
            'title',
            'message',
            'timestamp',
            'resource',
            'seen'
        )
