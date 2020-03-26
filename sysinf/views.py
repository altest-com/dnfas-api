from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from . import services


class SystemStatsView(APIView):

    def get(self, request):

        all_stats = [
            'cpu',
            'gpu',
            'memory',
            'disk',
            'network',
            'temperature'
        ]

        stats = self.request.query_params.getlist('stats', None)
        if stats is None or len(stats) == 0:
            stats = all_stats

        data = {}
        for stat in stats:
            if stat in all_stats:
                stat_func = getattr(services, stat, None)
                if stat_func is not None:
                    data[stat] = stat_func()

        return Response(data, status=status.HTTP_200_OK)