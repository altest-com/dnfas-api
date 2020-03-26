from rest_framework import status, viewsets, mixins
from rest_framework.response import Response

from .mixins import (
    ListMixin
)
from ..models import Stat
from ..serializers import StatSerializer


class StatView(
    ListMixin,
    viewsets.GenericViewSet
):
    """
    list:
        Return all stats.
    """

    model_name = 'Stat'
    lookup_field = 'pk'
    queryset = Stat.objects.all()
    serializer_class = StatSerializer

    def get_queryset(self):
        queryset = self.queryset

        name = self.request.query_params.get('name', None)
        if name is not None:
            queryset = queryset.filter(name=name)

        resolution = self.request.query_params.get('resolution', None)
        if resolution is not None:
            queryset = queryset.filter(resolution=resolution)

        order_by = self.request.query_params.get('order_by', None)
        if order_by is not None:
            queryset = queryset.order_by(order_by)

        return queryset
