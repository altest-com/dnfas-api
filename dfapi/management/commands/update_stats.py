from django.core.management.base import BaseCommand
from dfapi.models import stat
from dfapi import services


class Command(BaseCommand):
    help = 'Update statistics'

    def _run(self):
        services.stats.update_time_stats(stat.Stat.RESOLUTION_DAY)
        services.stats.update_time_stats(stat.Stat.RESOLUTION_HOUR)
        services.stats.update_total_stats()

    def handle(self, *args, **options):
        self._run()