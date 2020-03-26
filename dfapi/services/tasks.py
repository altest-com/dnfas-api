import logging
from datetime import timedelta, datetime, time

from django.conf import settings
from django.db.models import Count
from django.utils.timezone import make_aware
from django.utils import timezone

from .exceptions import ServiceError
from .workers import RunnerManager, WorkerApi
from ..models import Task
from ..models import Worker

logger_name = settings.LOGGER_NAME
logger = logging.getLogger(logger_name)

CHECK_TASKS_MAX_AGE_DAYS = 7
CREATED_TASK_TIMEOUT_DAYS = 1

runner_manager = RunnerManager()


def select_worker():
    active_status = (
        Task.STATUS_CREATED,
        Task.STATUS_RUNNING,
        Task.STATUS_PAUSED
    )
    queryset = Worker.objects.exclude(tasks__status__in=active_status)
    if len(queryset):
        for worker in queryset:
            if worker.is_self():
                return worker
            worker_api = WorkerApi(api_url=worker.api_url)
            if worker_api.is_online():
                return worker
            else:
                logger.warning(f'Worker at {worker.api_url} is offline.')

    queryset = Worker.objects.filter(
        tasks__status__in=active_status
    ).annotate(
        tasks_count=Count('tasks')
    ).order_by('tasks_count')

    if not len(queryset):
        return None

    for worker in queryset:
        if worker.tasks_count < worker.max_load:
            if worker.is_self():
                return worker
            worker_api = WorkerApi(api_url=worker.api_url)
            if worker_api.is_online():
                return worker
            else:
                logger.warning(f'Worker at {worker.api_url} is offline.')

    return None


def create(task: Task):
    datetime_now = make_aware(datetime.now())
    if task.schedule_start_at is None or (
        task.schedule_start_at is not None and
        datetime_now >= task.schedule_start_at
    ):
        start_task(task)


def start_task(task: Task):

    worker: Worker = select_worker()
    if worker is None:
        return

    task.worker = worker
    task.save(update_fields=['worker'])

    if worker.is_self():
        runner_manager.create(task.pk)
    else:
        worker_api = WorkerApi(api_url=worker.api_url)
        worker_api.execute(
            resource=task.pk,
            action=WorkerApi.ACTION_START,
            username=worker.username,
            password=worker.password
        )


def pause_task(task: Task):
    worker: Worker = task.worker

    if worker is None:
        raise ServiceError(f'Can not pause task <{task.pk}> because its worker is undefined.')

    if worker.name.lower() == settings.WORKER_NAME.lower():
        runner_manager.pause(task.pk)
    else:
        worker_api = WorkerApi(api_url=worker.api_url)
        worker_api.execute(
            resource=task.pk,
            action=WorkerApi.ACTION_PAUSE,
            username=worker.username,
            password=worker.password
        )


def resume_task(task):
    worker: Worker = task.worker

    if worker is None:
        raise ServiceError(f'Can not resume task <{task.pk}> because its worker is undefined.')

    if worker.name.lower() == settings.WORKER_NAME.lower():
        runner_manager.resume(task.pk)
    else:
        worker_api = WorkerApi(api_url=worker.api_url)
        worker_api.execute(
            resource=task.pk,
            action=WorkerApi.ACTION_RESUME,
            username=worker.username,
            password=worker.password
        )


def stop_task(task):
    worker: Worker = task.worker

    if worker is None:
        raise ServiceError(f'Can not stop task <{task.pk}> because its worker is undefined.')

    if worker.name.lower() == settings.WORKER_NAME.lower():
        runner_manager.stop(task.pk)
    else:
        worker_api = WorkerApi(api_url=worker.api_url)
        worker_api.execute(
            resource=task.pk,
            action=WorkerApi.ACTION_STOP,
            username=worker.username,
            password=worker.password
        )


def schedule_tasks():

    now = make_aware(datetime.now())
    min_timestamp = now - timedelta(days=CHECK_TASKS_MAX_AGE_DAYS)
    tasks = Task.objects.filter(
        updated_at__gt=min_timestamp,
        repeat_days__exact=''
    )

    for task in tasks:
        if task.status in (Task.STATUS_RUNNING, Task.STATUS_PAUSED):
            if (
                task.schedule_stop_at is not None and
                now > task.schedule_stop_at
            ):
                try:
                    stop_task(task)
                except ServiceError as err:
                    logger.error(err)
        else:
            if (
                task.schedule_start_at is not None and
                now > task.schedule_start_at and
                task.started_at is None
            ):
                try:
                    start_task(task)
                except ServiceError as err:
                    logger.error(err)


def repeat_tasks():

    now = timezone.localtime(timezone.now())
    date_now = now.date()
    time_now = now.time()

    weekday = str(date_now.weekday())

    # Get tasks that must run today
    tasks = Task.objects.filter(
        repeat_days__icontains=weekday
    )
    time_max_stop = time(hour=23)

    for task in tasks:
        schedule_stop_at = time_max_stop
        if task.schedule_stop_at is not None:
            schedule_stop_at = timezone.localtime(
                task.schedule_stop_at).time()

        schedule_start_at = None
        if task.schedule_start_at is not None:
            schedule_start_at = timezone.localtime(
                task.schedule_start_at).time()

        if task.status in (Task.STATUS_RUNNING, Task.STATUS_PAUSED):
            if time_now > schedule_stop_at:
                try:
                    stop_task(task)
                except ServiceError as err:
                    logger.error(err)
        else:
            run_today = (
                task.started_at is not None and
                timezone.localdate(task.started_at) == date_now
            )

            if not run_today and (schedule_start_at is None or (
                schedule_start_at is not None and
                time_now >= schedule_start_at
            )):
                try:
                    start_task(task)
                except ServiceError as err:
                    logger.error(err)
