from rest_framework import viewsets

from .mixins import (
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin
)
from ..models import Tag
from ..serializers import TagSerializer


class TagView(
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin,
    viewsets.GenericViewSet
):
    """
    retrieve:
        Return a frame instance.

    list:
        Return all frames.

    create:
        Create a new frame.

    destroy:
        Remove an existing frame.

    partial_update:
        Update one or more fields on an existing frame.

    detect_faces:
        Detect faces in a frame instance.
    """

    model_name = 'Tag'
    lookup_field = 'pk'
    queryset = Tag.objects.all()
    serializer_class = TagSerializer

    def get_queryset(self):
        queryset = self.queryset
        params = self.request.query_params

        name = params.get('name', None)
        if name is not None:
            queryset = queryset.filter(name__icontains=name)

        model = params.get('model', None)
        if model is not None:
            queryset = queryset.filter(model__exact=model)

        if params.get('order_by', None) is not None:
            queryset = queryset.order_by(params['order_by'])

        return queryset
