from datetime import datetime

from django.conf import settings
from django.http import HttpResponse
from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from ..services.subjects import xls_export

from .mixins import (
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin
)
from .. import services
from ..models import Subject
from ..models import SubjectSegment
from ..serializers import (
    SubjectSerializer,
    SubjectSegmentSerializer
)

XLS_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
XLS_NAME = 'attachment; filename="faces-{}.xlsx"'


class SubjectView(
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin,
    viewsets.GenericViewSet
):
    """
    retrieve:
        Return a subject instance.

    list:
        Return all subjects.

    create:
        Create a new subject.

    destroy:
        Remove an existing subject.

    partial_update:
        Update one or more fields on an existing subject.
    """

    model_name = 'Subject'
    lookup_field = 'pk'
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer

    def get_queryset(self):
        queryset, _ = services.subjects.build_queryset(
            self.queryset, self.request.query_params
        )
        return queryset


class DemograpView(
    ListMixin,
    viewsets.GenericViewSet
):
    """
    list:
        Return all subjects.
    """

    model_name = 'Subject'
    lookup_field = 'pk'
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer

    def get_queryset(self):
        queryset = self.queryset.exclude(pred_sex__exact='')
        queryset = queryset.exclude(pred_age__isnull=True)

        queryset, _ = services.subjects.build_queryset(
            queryset, self.request.query_params
        )
        return queryset

    @action(detail=False, methods=['get'])
    def stats(self, request):
        data = services.subjects.demograp(self.get_queryset())
        return Response(data, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'])
    def export(self, request, *args, **kwargs):
        now = datetime.now().strftime('%Y-%m-%d')
        response = HttpResponse(content_type=XLS_MIME)
        response['Content-Disposition'] = XLS_NAME.format(now)

        columns = self.request.query_params.getlist('columns', None)
        workbook = xls_export(
            self.get_queryset(),
            columns=columns,
            request=request
        )
        workbook.save(response)

        return response


class SubjectSegmentView(
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin,
    viewsets.GenericViewSet
):
    """
    retrieve:
        Return a subject segment instance.

    list:
        Return all subject segments.

    create:
        Create a new subject segment.

    destroy:
        Remove an existing subject segment.

    partial_update:
    """

    model_name = 'SubjectSegment'
    lookup_field = 'pk'
    queryset = SubjectSegment.objects.exclude(
        title=settings.DEFAULT_SEGMENT_TITLE
    ).filter(disk_cached=True)

    serializer_class = SubjectSegmentSerializer

