from datetime import datetime, timedelta

from django.db.models import Sum
from django.utils.timezone import make_aware

from ..models import Face, Subject, Task, Frame, VideoRecord, Camera, Stat


def update_total_stats():

    def faces_image_size():
        result = Face.objects.all().aggregate(
            Sum('size_bytes'))['size_bytes__sum']
        return result if result is not None else 0

    def frames_image_size():
        result = Frame.objects.all().aggregate(
            Sum('size_bytes'))['size_bytes__sum']
        return result if result is not None else 0

    def videos_size():
        result = VideoRecord.objects.all().aggregate(
            Sum('size_bytes'))['size_bytes__sum']
        return result if result is not None else 0

    def stored_subjects_eval():
        # return randint(0, 50)
        result = Subject.objects.filter().count()
        return result

    def stored_faces_eval():
        # return randint(0, 50)
        result = Face.objects.filter().count()
        return result

    def stored_frames_eval():
        # return randint(0, 50)
        result = Frame.objects.filter().count()
        return result

    def stored_cameras_eval():
        # return randint(0, 50)
        result = Camera.objects.filter().count()
        return result

    def stored_videos_eval():
        # return randint(0, 50)
        result = VideoRecord.objects.filter().count()
        return result

    def stored_tasks_eval():
        # return randint(0, 50)
        result = Task.objects.filter().count()
        return result

    stats_kwargs = [
        {
            'name': 'faces_image_size',
            'value_eval': faces_image_size
        }, {
            'name': 'frames_image_size',
            'value_eval': frames_image_size
        }, {
            'name': 'videos_size',
            'value_eval': videos_size
        }, {
            'name': 'stored_subjects',
            'value_eval': stored_subjects_eval
        }, {
            'name': 'stored_faces',
            'value_eval': stored_faces_eval
        }, {
            'name': 'stored_frames',
            'value_eval': stored_frames_eval
        }, {
            'name': 'stored_cameras',
            'value_eval': stored_cameras_eval
        }, {
            'name': 'stored_videos',
            'value_eval': stored_videos_eval
        }, {
            'name': 'stored_tasks',
            'value_eval': stored_tasks_eval
        }
    ]

    for stat_kwargs in stats_kwargs:
        stat, _ = Stat.objects.get_or_create(
            name=stat_kwargs['name'],
            resolution=Stat.RESOLUTION_ALL
        )
        stat.timestamp = make_aware(datetime.now())
        stat.value = stat_kwargs['value_eval']()
        stat.save()


def update_time_stats(resolution):

    def faces_count_eval(min_timestamp, max_timestamp):
        # return randint(0, 100)
        result = Task.objects.filter(
            created_at__gt=min_timestamp,
            created_at__lte=max_timestamp
        ).aggregate(Sum('faces_count'))['faces_count__sum']
        return result if result is not None else 0

    def frames_count_eval(min_timestamp, max_timestamp):
        # return randint(0, 1000)
        result = Task.objects.filter(
            created_at__gt=min_timestamp,
            created_at__lte=max_timestamp
        ).aggregate(Sum('frames_count'))['frames_count__sum']
        return result if result is not None else 0

    def processing_time_eval(min_timestamp, max_timestamp):
        # return 7200 * random() + 1800
        result = Task.objects.filter(
            created_at__gt=min_timestamp,
            created_at__lte=max_timestamp
        ).aggregate(Sum('processing_time'))['processing_time__sum']
        return result if result is not None else 0

    def tasks_count_eval(min_timestamp, max_timestamp):
        # return 7200 * random() + 1800
        result = Task.objects.filter(
            created_at__gt=min_timestamp,
            created_at__lte=max_timestamp,
            status__in=[Task.STATUS_SUCCESS, Task.STATUS_STOPPED, Task.STATUS_KILLED]
        ).distinct().count()
        return result if result is not None else 0

    stats_kwargs = [
        {
            'name': 'faces_count',
            'value_eval': faces_count_eval
        }, {
            'name': 'frames_count',
            'value_eval': frames_count_eval
        }, {
            'name': 'processing_time',
            'value_eval': processing_time_eval
        }, {
            'name': 'tasks_count',
            'value_eval': tasks_count_eval
        }
    ]

    for stat_kwargs in stats_kwargs:
        update_stat(resolution=resolution, **stat_kwargs)


def update_stat(name: str, resolution: str, value_eval: callable):

    if resolution == Stat.RESOLUTION_HOUR:
        replace_kwargs = {
            'minute': 0,
            'second': 0,
            'microsecond': 0
        }
        backward_kwargs = {'hours': 24}
        forward_kwargs = {'hours': 1}
    elif resolution == Stat.RESOLUTION_DAY:
        replace_kwargs = {
            'hour': 0,
            'minute': 0,
            'second': 0,
            'microsecond': 0
        }
        backward_kwargs = {'days': 30}
        forward_kwargs = {'hours': 24}
    else:
        raise ValueError(f'Invalid resolution "{resolution}"')

    query = Stat.objects.filter(
        name=name,
        resolution=resolution
    ).order_by('-timestamp')

    last_stat = None
    if query.exists():
        last_stat = query.last()

    now = make_aware(datetime.now())
    now = now.replace(**replace_kwargs)
    min_timestamp = now - timedelta(**backward_kwargs)

    if not last_stat:
        last_update_at = min_timestamp
    else:
        last_update_at = last_stat.timestamp.replace(**replace_kwargs)
        if last_update_at < min_timestamp:
            last_update_at = min_timestamp
            stats_query = Stat.objects.filter(
                name=name,
                resolution=resolution,
                timestamp__lte=min_timestamp
            )
            if stats_query.exists():
                stats_query.delete()

    # count = Face.objects.filter(
    #     created_at__gt=last_update_at
    # ).count()
    #
    # if count == 0:
    #     return

    timestamp_prev = last_update_at
    timestamp = last_update_at + timedelta(**forward_kwargs)

    while timestamp <= now:

        value = value_eval(min_timestamp=timestamp_prev, max_timestamp=timestamp)
        # value = Face.objects.filter(
        #     created_at__gt=timestamp_prev,
        #     created_at__lte=timestamp
        # ).count()

        stat, _ = Stat.objects.get_or_create(
            name=name,
            resolution=resolution,
            timestamp=timestamp
        )

        stat.value = value
        stat.save()

        timestamp_prev = timestamp
        timestamp = timestamp + timedelta(**forward_kwargs)