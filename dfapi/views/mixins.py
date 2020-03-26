from django.core.exceptions import (
    ObjectDoesNotExist,
    FieldError,
    ValidationError
)
from rest_framework import status
from rest_framework.exceptions import (
    NotFound as ApiNotFound,
    ValidationError as ApiValidationError
)
from rest_framework.response import Response


class RetrieveMixin:

    # noinspection PyUnresolvedReferences
    def retrieve(self, request, pk):
        serializer_context = {'request': request}
        try:
            pk = int(pk)
            instance = self.queryset.get(pk=pk)
        except (ObjectDoesNotExist, ValueError):
            raise ApiNotFound(f'A {self.model_name} with pk={pk} does not exist.')

        serializer = self.serializer_class(
            instance,
            context=serializer_context
        )
        return Response(serializer.data, status=status.HTTP_200_OK)


class ListMixin:

    # noinspection PyUnresolvedReferences
    def list(self, request):
        try:
            list_serializer_class = self.list_serializer_class
        except AttributeError:
            list_serializer_class = self.serializer_class
        serializer_context = {'request': request}

        try:
            queryset = self.get_queryset()
        except ValidationError:
            raise ApiValidationError(f'Invalid query parameters.')

        try:
            page = self.paginate_queryset(queryset)
        except FieldError:
            raise ApiValidationError(f'Invalid query parameters.')

        fields = request.query_params.get('fields', None)
        if fields is not None:
            fields = fields.split(',')

        serializer = list_serializer_class(
            page,
            context=serializer_context,
            many=True,
            fields=fields
        )
        return self.get_paginated_response(serializer.data)


class CreateMixin:

    # noinspection PyUnresolvedReferences
    def create(self, request):
        try:
            create_serializer_class = self.create_serializer_class
        except AttributeError:
            create_serializer_class = self.serializer_class
        serializer_context = {'request': request}
        serializer = create_serializer_class(
            data=request.data,
            context=serializer_context
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        serializer = self.serializer_class(
            serializer.instance,
            context=serializer_context
        )
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class UpdateMixin:

    # noinspection PyUnresolvedReferences
    def partial_update(self, request, pk=None):
        try:
            update_serializer_class = self.update_serializer_class
        except AttributeError:
            update_serializer_class = self.serializer_class
        try:
            pk = int(pk)
            instance = self.queryset.get(pk=pk)
        except (ObjectDoesNotExist, ValueError):
            raise ApiNotFound(f'A {self.model_name} with pk={pk} does not exist.')

        serializer_context = {'request': request}
        serializer = update_serializer_class(
            instance,
            data=request.data,
            context=serializer_context,
            partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        serializer = self.serializer_class(
            serializer.instance,
            context=serializer_context
        )
        return Response(serializer.data, status=status.HTTP_200_OK)


class DestroyMixin:

    # noinspection PyUnresolvedReferences
    def destroy(self, request, pk=None):
        try:
            pk = int(pk)
            instance = self.queryset.get(pk=pk)
        except (ObjectDoesNotExist, ValueError):
            raise ApiNotFound(f'A {self.model_name} with pk={pk} does not exist.')

        instance.delete()

        return Response(None, status=status.HTTP_204_NO_CONTENT)
