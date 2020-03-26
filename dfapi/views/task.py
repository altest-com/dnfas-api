from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import NotFound, ValidationError
from rest_framework.response import Response

from .mixins import (
    RetrieveMixin,
    ListMixin
)
from .. import services
from ..models import Task
from ..serializers import TaskSerializer


class TaskView(
    RetrieveMixin,
    ListMixin,
    viewsets.GenericViewSet,
):
    """
    retrieve:
        Return a task instance.

    list:
        Return all task.

    create:
        Create a new task.

    destroy:
        Remove an existing task.

    start:
        Start task execution.

    pause:
        Start task execution.

    resume:
        Resume task execution.

    stop:
        Stop task execution.
    """

    model_name = 'Task'
    lookup_field = 'pk'
    queryset = Task.objects.all()
    serializer_class = TaskSerializer

    def get_queryset(self):
        queryset = self.queryset
        params = self.request.query_params

        name = params.get('name', None)
        if name is not None:
            queryset = queryset.filter(name__icontains=name)

        task_type = params.getlist('task_type', None)
        if task_type is not None and len(task_type):
            queryset = queryset.filter(task_type__in=task_type)

        task_status = params.getlist('status', None)
        if task_status is not None and len(task_status):
            queryset = queryset.filter(status__in=task_status)
        # elif params.get('status', None) is not None:
        #     queryset = queryset.filter(status=params['status'])

        if params.get('order_by', None) is not None:
            queryset = queryset.order_by(params['order_by'])

        return queryset

    def create(self, request):

        serializer_context = {'request': request}
        serializer = self.serializer_class(
            data=request.data,
            context=serializer_context
        )
        serializer.is_valid(raise_exception=True)
        task = serializer.save()

        try:
            services.tasks.create(task)
        except services.ServiceError as err:
            raise ValidationError(err)

        serializer = self.serializer_class(
            task,
            context=serializer_context
        )

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def partial_update(self, request, pk=None):

        try:
            pk = int(pk)
            task: Task = self.queryset.get(pk=pk)
        except (Task.DoesNotExist, ValueError):
            raise NotFound(f'A task with pk={pk} does not exist.')

        if task.status in (Task.STATUS_RUNNING, Task.STATUS_PAUSED):
            raise ValidationError(f'Attempt to update an active task.')

        serializer_context = {'request': request}
        serializer = self.serializer_class(
            task,
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

    def destroy(self, request, pk):

        try:
            pk = int(pk)
            task = self.queryset.get(pk=pk)
        except (Task.DoesNotExist, ValueError):
            raise NotFound(f'A task with pk={pk} does not exist.')

        if task.status in (Task.STATUS_RUNNING, Task.STATUS_PAUSED):
            raise ValidationError(f'Attempt to delete an active task.')

        try:
            services.tasks.stop_task(task)
        except services.ServiceError as err:
            pass

        task.delete()

        return Response(None, status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=['patch'])
    def start(self, request, pk):
        return self._do_action(request, pk, 'start')

    @action(detail=True, methods=['patch'])
    def pause(self, request, pk):
        return self._do_action(request, pk, 'pause')

    @action(detail=True, methods=['patch'])
    def resume(self, request, pk):
        return self._do_action(request, pk, 'resume')

    @action(detail=True, methods=['patch'])
    def stop(self, request, pk):
        return self._do_action(request, pk, 'stop')

    def _do_action(self, request, pk, action_name):
        serializer_context = {'request': request}

        try:
            pk = int(pk)
            task = Task.objects.get(pk=pk)
        except (Task.DoesNotExist, ValueError):
            raise NotFound(f'A task with pk={pk} does not exists.')

        try:
            if action_name == 'start':
                services.tasks.start_task(task)
            elif action_name == 'pause':
                services.tasks.pause_task(task)
            elif action_name == 'resume':
                services.tasks.resume_task(task)
            elif action_name == 'stop':
                services.tasks.stop_task(task)
        except services.ServiceError as err:
            raise ValidationError(err)

        serializer = self.serializer_class(
            task,
            context=serializer_context
        )
        return Response(serializer.data, status=status.HTTP_200_OK)