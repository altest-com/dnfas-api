# Generated by Django 3.0.2 on 2020-02-17 18:24

import django.contrib.postgres.fields.jsonb
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dfapi', '0003_auto_20200108_1532'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='task',
            name='camera',
        ),
        migrations.RemoveField(
            model_name='task',
            name='detection_min_height',
        ),
        migrations.RemoveField(
            model_name='task',
            name='detection_min_score',
        ),
        migrations.RemoveField(
            model_name='task',
            name='faces_count',
        ),
        migrations.RemoveField(
            model_name='task',
            name='faces_time_memory',
        ),
        migrations.RemoveField(
            model_name='task',
            name='frame_rate',
        ),
        migrations.RemoveField(
            model_name='task',
            name='frames_count',
        ),
        migrations.RemoveField(
            model_name='task',
            name='frontal_faces',
        ),
        migrations.RemoveField(
            model_name='task',
            name='hunted_subjects',
        ),
        migrations.RemoveField(
            model_name='task',
            name='max_frame_size',
        ),
        migrations.RemoveField(
            model_name='task',
            name='mode',
        ),
        migrations.RemoveField(
            model_name='task',
            name='processing_time',
        ),
        migrations.RemoveField(
            model_name='task',
            name='similarity_thresh',
        ),
        migrations.RemoveField(
            model_name='task',
            name='store_face_frames',
        ),
        migrations.RemoveField(
            model_name='task',
            name='video',
        ),
        migrations.RemoveField(
            model_name='task',
            name='video_detect_interval',
        ),
        migrations.RemoveField(
            model_name='task',
            name='video_start_at',
        ),
        migrations.RemoveField(
            model_name='task',
            name='video_stop_at',
        ),
        migrations.AddField(
            model_name='face',
            name='pred_age',
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='face',
            name='pred_sex',
            field=models.CharField(blank=True, choices=[('man', 'man'), ('women', 'woman')], default='', max_length=16),
        ),
        migrations.AddField(
            model_name='task',
            name='config',
            field=django.contrib.postgres.fields.jsonb.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='task',
            name='info',
            field=django.contrib.postgres.fields.jsonb.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='task',
            name='name',
            field=models.CharField(default='Task', max_length=255),
        ),
        migrations.AddField(
            model_name='task',
            name='task_type',
            field=models.CharField(choices=[('video_detect_faces', 'video_detect_faces'), ('video_hunt_faces', 'video_hunt_faces'), ('video_detect_person', 'video_detect_person'), ('video_hunt_person', 'video_hunt_person'), ('predict_genderage', 'predict_genderage')], default='video_detect_faces', max_length=64),
        ),
        migrations.AlterField(
            model_name='task',
            name='repeat_days',
            field=models.CharField(blank=True, default='0000000', max_length=7),
        ),
    ]
