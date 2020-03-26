from django.core.management.base import BaseCommand
from dfapi.models import Face, Subject, Frame, VideoRecord, VideoThumb

models = [
    Face,
    Subject,
    Frame,
    VideoRecord,
    VideoThumb
]


class Command(BaseCommand):
    help = 'Clean the database'

    def handle(self, *args, **options):
        for model in models:
            queryset = model.objects.all()
            print(f'Deleting {len(queryset)} <{repr(model)}> instances...')
            queryset.delete()
            print(f'All instances deleted.')