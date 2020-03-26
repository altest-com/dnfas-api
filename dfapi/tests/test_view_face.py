from rest_framework import status
from rest_framework.reverse import reverse
from rest_framework.test import APITransactionTestCase

from .factory import (
    ModelFactory,
    FaceFactory,
    SubjectFactory,
    SubjectSegmentFactory,
    RecognitionFactory,
    NotificationFactory,
    CameraFactory,
    VideoFactory,
    FrameFactory,
    StatFactory
)


class _ViewTest(APITransactionTestCase):

    url_list = ''
    url_detail = ''
    model_factory: ModelFactory = None
    list_count = 1
    valid_data = {}
    list_instances = []
    destroy_instances = []

    def setUp(self):
        self.invalid_data = {}

        self.valid_data = {
            'Full data': self.model_factory.api_post_data(),
            'Minimal data': self.model_factory.api_post_data(full=False)
        }

        self.instances = [
            self.model_factory.create_instance()
            for _ in range(self.list_count)
        ]

    def tearDown(self):
        instances = self.model_factory.model_cls.objects.all()
        for instance in instances:
            instance.delete()


# noinspection PyUnresolvedReferences
class _MixinViewCreateTest:

    def test_create(self):
        for label, data in self.valid_data.items():
            with self.subTest(msg=label):
                response = self.client.post(
                    reverse(self.url_list),
                    data=data
                )
                self.assertEqual(
                    status.HTTP_201_CREATED,
                    response.status_code,
                    msg=repr(response.data)
                )


# noinspection PyUnresolvedReferences
class _MixinViewListTest:

    def test_list(self):
        response = self.client.get(reverse(self.url_list))
        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK,
            msg=repr(response.data)
        )
        data = response.data['results']
        for index, item in enumerate(data):
            with self.subTest(msg=f'List index {index}'):
                self.assertSetEqual(
                    set(self.model_factory.API_READ_FIELDS),
                    set(item.keys())
                )


# noinspection PyUnresolvedReferences
class _MixinViewDeleteTest:

    def test_delete(self):
        for instance in self.instances:
            with self.subTest(msg=f'Model pk={instance.pk}'):
                response = self.client.delete(
                    reverse(self.url_detail, kwargs={'pk': instance.pk})
                )
                self.assertEqual(
                    response.status_code,
                    status.HTTP_204_NO_CONTENT,
                    msg=repr(response.data)
                )


# noinspection PyUnresolvedReferences
class _MixinViewRetrieveTest:

    def test_retrieve(self):
        for instance in self.instances:
            with self.subTest(msg=f'Model pk={instance.pk}'):
                response = self.client.get(
                    reverse(self.url_detail, kwargs={'pk': instance.pk})
                )
                self.assertEqual(
                    response.status_code,
                    status.HTTP_200_OK,
                    msg=repr(response.data)
                )
                self.assertSetEqual(
                    set(self.model_factory.API_READ_FIELDS),
                    set(response.data.keys())
                )


# noinspection PyUnresolvedReferences
class _MixinViewUpdateTest:

    def test_update(self):
        instance = self.instances[0]
        for label, data in self.valid_data.items():
            with self.subTest(msg=label):
                response = self.client.patch(
                    reverse(self.url_detail, kwargs={'pk': instance.pk}),
                    data=data
                )
                self.assertEqual(
                    status.HTTP_200_OK,
                    response.status_code,
                    msg=repr(response.data)
                )
                self.assertSetEqual(
                    set(self.model_factory.API_READ_FIELDS),
                    set(response.data.keys())
                )


class FaceViewTest(
    _ViewTest,
    _MixinViewCreateTest,
    _MixinViewListTest,
    _MixinViewRetrieveTest,
    _MixinViewUpdateTest,
    _MixinViewDeleteTest
):

    url_list = 'dfapi:faces-list'
    url_detail = 'dfapi:faces-detail'
    model_factory = FaceFactory()


class FrameViewTest(
    _ViewTest,
    _MixinViewCreateTest,
    _MixinViewListTest,
    _MixinViewRetrieveTest,
    _MixinViewUpdateTest,
    _MixinViewDeleteTest
):

    url_list = 'dfapi:frames-list'
    url_detail = 'dfapi:frames-detail'
    url_detect_faces = 'dfapi:frames-detect-faces'
    model_factory = FrameFactory()

    def test_detect_faces(self):
        instance = self.instances[0]
        response = self.client.post(
            reverse(self.url_detect_faces, kwargs={'pk': instance.pk})
        )
        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK,
            msg=repr(response.data)
        )

        for index, item in enumerate(response.data):
            with self.subTest(msg=f'List index {index}'):
                self.assertSetEqual(
                    set(FaceFactory.API_READ_FIELDS),
                    set(item.keys())
                )


class VideoViewTest(
    _ViewTest,
    _MixinViewCreateTest,
    _MixinViewListTest,
    _MixinViewRetrieveTest,
    _MixinViewUpdateTest,
    _MixinViewDeleteTest
):

    url_list = 'dfapi:videos-list'
    url_detail = 'dfapi:videos-detail'
    model_factory = VideoFactory()
    list_count = 1


class CameraViewTest(
    _ViewTest,
    _MixinViewCreateTest,
    _MixinViewListTest,
    _MixinViewRetrieveTest,
    _MixinViewUpdateTest,
    _MixinViewDeleteTest
):

    url_list = 'dfapi:cameras-list'
    url_detail = 'dfapi:cameras-detail'
    model_factory = CameraFactory()


class SubjectViewTest(
    _ViewTest,
    _MixinViewCreateTest,
    _MixinViewListTest,
    _MixinViewRetrieveTest,
    _MixinViewUpdateTest,
    _MixinViewDeleteTest
):

    url_list = 'dfapi:subjects-list'
    url_detail = 'dfapi:subjects-detail'
    model_factory = SubjectFactory()


class SubjectSegmentViewTest(
    _ViewTest,
    _MixinViewCreateTest,
    _MixinViewListTest,
    _MixinViewRetrieveTest,
    _MixinViewUpdateTest,
    _MixinViewDeleteTest
):

    url_list = 'dfapi:segments-list'
    url_detail = 'dfapi:segments-detail'
    model_factory = SubjectSegmentFactory()
    list_count = 1


class NotificationViewTest(
    _ViewTest,
    _MixinViewListTest,
    _MixinViewDeleteTest
):

    url_list = 'dfapi:notifications-list'
    url_detail = 'dfapi:notifications-detail'
    url_see = 'dfapi:notifications-see'
    model_factory = NotificationFactory()

    def test_see(self):
        instance = self.instances[0]
        response = self.client.post(
            reverse(self.url_see, kwargs={'pk': instance.pk})
        )
        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK,
            msg=repr(response.data)
        )

        self.assertSetEqual(
            set(response.data),
            set(NotificationFactory.API_READ_FIELDS)
        )


class RecognitionViewTest(
    _ViewTest,
    _MixinViewCreateTest,
    _MixinViewListTest,
    _MixinViewRetrieveTest,
    _MixinViewDeleteTest
):

    url_list = 'dfapi:recognitions-list'
    url_detail = 'dfapi:recognitions-detail'
    model_factory = RecognitionFactory()
    list_count = 1


class StatViewTest(
    _ViewTest,
    _MixinViewListTest
):

    url_list = 'dfapi:stats-list'
    model_factory = StatFactory()

# from django.test import TestCase, TransactionTestCase
# from multiprocessing import Process
# from time import sleep
# from django import db
#
# def execute_task():
#     print('Process started')
#     sleep(5)
#     print(Face.objects.all())
#
#
# class BugsTest(TransactionTestCase):
#
#     def test_multiprocessing(self):
#         face_1 = FaceFactory.create_instance()
#         print('Face 1 created')
#         db.connections.close_all()
#         process = Process(
#             target=execute_task
#         )
#         process.start()
#         face_2 = FaceFactory.create_instance()
#         print('Face 2 created')
#         sleep(10)
