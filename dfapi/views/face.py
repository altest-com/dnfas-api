from rest_framework import viewsets

from .mixins import (
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin
)
from ..models import Face
from ..serializers import FaceSerializer


class FaceView(
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin,
    viewsets.GenericViewSet
):
    """
    retrieve:
        Return a face instance.

    list:
        Return all faces.

    create:
        Create a new face.

    destroy:
        Remove an existing face.

    partial_update:
        Update one or more fields on an existing face.
    """

    model_name = 'Face'
    lookup_field = 'pk'
    queryset = Face.objects.all()
    serializer_class = FaceSerializer

    def get_queryset(self):
        queryset = self.queryset

        camera = self.request.query_params.get('camera', None)
        if camera is not None:
            queryset = queryset.filter(task__camera=camera)

        order_by = self.request.query_params.get('order_by', None)
        if order_by is not None:
            queryset = queryset.order_by(order_by)

        return queryset
