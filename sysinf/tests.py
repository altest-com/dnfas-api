from rest_framework import serializers
from rest_framework import status
from rest_framework.reverse import reverse
from rest_framework.test import APITestCase


class GpuSerializer(serializers.Serializer):
    name = serializers.CharField()
    temperature = serializers.FloatField()
    fan_speed = serializers.FloatField(allow_null=True)
    utilization = serializers.FloatField()
    memory_used = serializers.FloatField()
    memory_total = serializers.FloatField()
    power_draw = serializers.FloatField()


class MemorySerializer(serializers.Serializer):
    total = serializers.FloatField()
    used = serializers.FloatField()
    free = serializers.FloatField()


class DiskSerializer(serializers.Serializer):
    total = serializers.FloatField()
    used = serializers.FloatField()
    free = serializers.FloatField()
    read_bytes = serializers.FloatField()
    write_bytes = serializers.FloatField()
    read_time = serializers.FloatField()
    write_time = serializers.FloatField()


class NetworkSerializer(serializers.Serializer):
    bytes_sent = serializers.FloatField()
    bytes_recv = serializers.FloatField()
    packets_sent = serializers.FloatField()
    packets_recv = serializers.FloatField()


class TemperatureSerializer(serializers.Serializer):
    label = serializers.CharField()
    value = serializers.FloatField()


class SystemStatSerializer(serializers.Serializer):
    cpu = serializers.ListSerializer(
        child=serializers.FloatField()
    )

    gpu = serializers.ListSerializer(
        child=GpuSerializer()
    )

    memory = MemorySerializer()

    disk = DiskSerializer()

    network = NetworkSerializer()

    temperature = serializers.ListSerializer(
        child=TemperatureSerializer()
    )


class ApiViewTest(APITestCase):

    url_list = 'sysinf:system'

    def test_get_system_stats(self):
        response = self.client.get(reverse(self.url_list))
        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK,
            msg=repr(response.data)
        )
        serial = SystemStatSerializer(data=response.data)
        serial.is_valid(raise_exception=True)
