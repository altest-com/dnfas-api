from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.reverse import reverse
from rest_framework.test import APITestCase

PASSWORD = '123'
USERNAME = 'lame'
EMAIL = 'lameboy@gmail.com'


def create_user(username=USERNAME, password=PASSWORD, email=EMAIL):
    return get_user_model().objects.create_user(
        username=username, password=password, email=email
    )


class AuthenticationTest(APITestCase):

    def test_user_can_login(self):
        user = create_user()
        response = self.client.post(reverse('users:login'), data={
            'username': USERNAME,
            'password': PASSWORD
        })
        self.assertEqual(status.HTTP_200_OK, response.status_code)
        self.assertEqual(response.data['username'], user.username)
        self.assertEqual(response.data['email'], user.email)
