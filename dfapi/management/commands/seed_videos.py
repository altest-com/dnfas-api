from django.core.management.base import BaseCommand
from dfapi.services.media import load_videos


class Command(BaseCommand):
    help = 'Create <VideoRecord> instances from files in source dir'

    def add_arguments(self, parser):
        # Positional arguments
        parser.add_argument('src', type=str)

    def handle(self, *args, **options):
        load_videos(dir_path=options['src'])