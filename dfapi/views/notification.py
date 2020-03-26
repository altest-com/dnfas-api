from django.core.exceptions import ObjectDoesNotExist
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import NotFound
from rest_framework.response import Response

from .mixins import (
    ListMixin,
    DestroyMixin
)
from ..models import Notification
from ..serializers import NotificationSerializer


class NotificationView(
    ListMixin,
    DestroyMixin,
    viewsets.GenericViewSet
):
    """
    list:
        Return all notifications.

    destroy:
        Remove an existing notification.

    see:
        Mark an existing notification as seen.
    """

    model_name = 'Notification'
    lookup_field = 'pk'
    queryset = Notification.objects.all()
    serializer_class = NotificationSerializer

    def get_queryset(self):
        queryset = self.queryset
        return queryset

    @action(detail=True, methods=['post'])
    def see(self, request, pk=None):
        serializer_context = {'request': request}

        try:
            pk = int(pk)
            notification = Notification.objects.get(pk=pk)
        except (ObjectDoesNotExist, ValueError):
            raise NotFound(f'Notification with pk={pk} does not exist.')

        notification.seen = True
        notification.save()

        serializer = self.serializer_class(
            notification,
            context=serializer_context
        )

        return Response(serializer.data, status=status.HTTP_200_OK)
