from os import path
from uuid import uuid4

import cv2 as cv
from cvtlib.files import list_files
from django.conf import settings
from django.core.management.base import BaseCommand

from dfapi.models import Subject, Frame
from dfapi.services.faces import face_analyzer
from warnings import  warn


# IMAGES_PATH = '/home/ronin/Projects/datasets/faces/cplfw/'
# IMAGE_EXTS = ('_1.jpg',)
# NMAX_SUBJECTS = 100


def create_image(file_path, media_path, prefix=''):
    image_name = f'{prefix}{uuid4()}.jpg'
    rel_path = path.join(media_path, image_name)
    full_path = path.join(settings.MEDIA_ROOT, rel_path)
    image = cv.imread(file_path)
    cv.imwrite(full_path, image)
    return rel_path


class Command(BaseCommand):
    help = 'Create subjects from faces images in directory'

    def add_arguments(self, parser):
        # Positional arguments
        parser.add_argument('src', type=str)

        # Named (optional) arguments
        parser.add_argument(
            '--exts',
            type=str,
            default='.jpg',
            help='Comma separated list of file extensions to read',
        )
        parser.add_argument(
            '--count',
            type=int,
            default=0,
            help='Maximum number of subjects to add',
        )

    def handle(self, *args, **options):
        images_path = options['src']
        image_exts = tuple([ext.strip() for ext in options['exts'].split(',')])
        max_subjects = options['count']

        if max_subjects <= 0:
            max_subjects = float('inf')

        print(f'Listing images from {images_path}...')
        image_paths = list_files(images_path, image_exts, recursive=True)

        files_count = len(image_paths)
        print(f'Found {files_count} images.')
        print(f'Starting processing images.')

        root = settings.FACES_IMAGES_PATH
        max_iter = min(files_count, max_subjects)

        for index, image_path in enumerate(image_paths):
            if index >= max_subjects:
                break
            image_name = path.basename(image_path)

            frame_path = create_image(image_path, root, 'frame_')
            frame = Frame.objects.create(image=frame_path)

            face_analyzer.analyze_frame(frame.pk)
            # frame = Frame.objects.get(pk=frame.pk)
            n_faces = frame.faces.count()
            if n_faces:
                face = frame.faces.all()[0]
                name = path.splitext(image_name)[0]
                subject_name = ' '.join(name.split('_')[0:-1])
                subject = Subject.objects.create(name=subject_name)
                subject.faces.add(face)
                if n_faces > 1:
                    warn(f'More than one face detected in "{image_name}".')
            else:
                warn(f'No face detected in "{image_name}".')

            print(f'Progress {(100 * index / max_iter):.1f}%.')

        print(f'Done!')
