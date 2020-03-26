from rest_framework import viewsets

from ..models import Camera, VideoRecord, Task
from ..serializers import CameraSerializer, VideoRecordSerializer
from .mixins import (
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin
)


class CameraView(
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin,
    viewsets.GenericViewSet
):
    """
    retrieve:
        Return a camera instance.

    list:
        Return all cameras.

    create:
        Create a new camera.

    destroy:
        Remove an existing camera.

    partial_update:
        Update one or more fields on an existing camera.
    """

    model_name = 'Camera'
    lookup_field = 'pk'
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer

    def get_queryset(self):
        queryset = self.queryset

        name = self.request.query_params.get('name', None)
        if name is not None:
            queryset = queryset.filter(name__icontains=name)

        address = self.request.query_params.get('address', None)
        if address is not None:
            queryset = queryset.filter(address__icontains=address)

        order_by = self.request.query_params.get('order_by', None)
        if order_by is not None:
            queryset = queryset.order_by(order_by)

        return queryset


class VideoRecordView(
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin,
    viewsets.GenericViewSet
):
    """
    retrieve:
        Return a video record instance.

    list:
        Return all video records.

    create:
        Create a new video record.

    destroy:
        Remove an existing video record.

    partial_update:
        Update one or more fields on an existing video record.
    """

    model_name = 'VideoRecord'
    lookup_field = 'pk'
    queryset = VideoRecord.objects.all()
    serializer_class = VideoRecordSerializer

    def get_queryset(self):
        queryset = self.queryset
        params = self.request.query_params

        name = self.request.query_params.get('name', None)
        if name is not None:
            queryset = queryset.filter(name__icontains=name)

        min_created_at = params.get('min_created_at', None)
        if min_created_at is not None:
            queryset = queryset.filter(created_at__gt=min_created_at)

        max_created_at = params.get('max_created_at', None)
        if max_created_at is not None:
            queryset = queryset.filter(created_at__lt=max_created_at)

        min_duration = params.get('min_duration', None)
        if min_duration is not None:
            try:
                queryset = queryset.filter(
                    duration_seconds__gt=float(min_duration)
                )
            except ValueError:
                pass

        max_duration = params.get('max_duration', None)
        if max_duration is not None:
            try:
                queryset = queryset.filter(
                    duration_seconds__lt=float(max_duration)
                )
            except ValueError:
                pass

        order_by = params.get('order_by', None)
        if order_by is not None:
            queryset = queryset.order_by(order_by)

        return queryset
