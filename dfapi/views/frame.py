from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import NotFound
from rest_framework.response import Response

from .mixins import (
    RetrieveMixin,
    ListMixin,
    CreateMixin,
    UpdateMixin,
    DestroyMixin
)
from ..models import Frame
from ..serializers import FrameSerializer
from ..services.faces import face_analyzer


class FrameView(
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

    model_name = 'Frame'
    lookup_field = 'pk'
    queryset = Frame.objects.all()
    serializer_class = FrameSerializer

    @action(detail=True, methods=['post'])
    def detect_faces(self, request, pk=None):

        try:
            pk = int(pk)
            frame = Frame.objects.get(pk=pk)
        except (Frame.DoesNotExist, ValueError):
            raise NotFound(f'A frame with pk={pk} does not exists.')

        for face in frame.faces.all():
            face.delete()

        face_analyzer.analyze_frame(frame.pk)
        # frame = Frame.objects.get(pk=pk)
        faces = [face.id for face in frame.faces.all()]

        return Response(faces, status=status.HTTP_200_OK)
