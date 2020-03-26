from django.contrib import admin


from .models import (
    Subject,
    SubjectSegment,
    Face,
    VideoRecord,
    Frame,
    VideoThumb,
    Camera,
    Task,
    Tag,
    Stat,
    Notification,
    Worker,
    Recognition,
    RecognitionMatch,
    HuntMatch
)


@admin.register(HuntMatch)
class HuntMatchAdmin(admin.ModelAdmin):
    pass


class RecognitionMatchInline(admin.TabularInline):
    model = RecognitionMatch
    fields = (
        'score', 'subject'
    )
    readonly_fields = (
        'score', 'subject'
    )


@admin.register(Recognition)
class RecognitionAdmin(admin.ModelAdmin):
    inlines = (RecognitionMatchInline,)


@admin.register(RecognitionMatch)
class RecognitionMatchAdmin(admin.ModelAdmin):
    pass


class FaceInline(admin.TabularInline):
    model = Face
    fields = (
        'id', 'frame', 'image'
    )
    readonly_fields = (
        'id', 'frame', 'image'
    )


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    fields = (
        'id',
        'name',
        'last_name',
        'birthdate',
        'full_name',
        'age',
        'sex',
        'pred_sex',
        'pred_sex_score',
        'pred_age',
        'pred_age_var',
        'skin',
        'created_at',
        'updated_at'
    )
    # list_display = (
    #     'id', 'name', 'created_at', 'updated_at',
    # )
    readonly_fields = (
        'id',
        'created_at',
        'updated_at',
        'full_name',
        'age'
    )
    inlines = (FaceInline,)


@admin.register(SubjectSegment)
class SubjectSegmentAdmin(admin.ModelAdmin):
    pass


@admin.register(Face)
class FaceAdmin(admin.ModelAdmin):
    fields = (
        'id',
        'frame',
        'image',
        'landmarks',
        'box',
        'created_at',
        'size_bytes',
        'subject',
        'pred_sex',
        'pred_sex_score',
        'pred_age',
        'pred_age_var',
        'embeddings'
    )
    readonly_fields = (
        'id',
        'landmarks',
        'box',
        'created_at',
        'pred_sex',
        'pred_age',
        'embeddings'
    )


@admin.register(VideoThumb)
class VideoThumbAdmin(admin.ModelAdmin):
    readonly_fields = (
        'id',
    )

@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    readonly_fields = (
        'id', 'created_at', 'updated_at'
    )


class VideoThumbInline(admin.TabularInline):
    model = VideoThumb
    readonly_fields = (
        'id', 'image'
    )


@admin.register(VideoRecord)
class VideoRecordAdmin(admin.ModelAdmin):

    fields = (
        'id',
        'name',
        'path',
        'created_at',
        'updated_at',
        'starts_at',
        'finish_at',
        'size',
        'size_bytes',
        'url',
        'frame_width',
        'frame_height',
        'duration_seconds',
        'full_path',
        'frames_count',
        'processing_time',
        'frame_rate'
    )
    readonly_fields = (
        'id',
        'created_at',
        'updated_at',
        'size',
        'url',
        'frame_width',
        'frame_height',
        'duration_seconds',
        'full_path',
        'frames_count',
        'processing_time',
        'frame_rate'
    )
    inlines = (VideoThumbInline,)


@admin.register(Frame)
class FrameAdmin(admin.ModelAdmin):
    fields = (
        'id', 'image', 'timestamp', 'size_bytes'
    )
    readonly_fields = (
        'id',
    )
    inlines = (FaceInline,)


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    fields = (
        'id',
        'stream_url',
        'name',
        'location_lat',
        'location_lon',
        'address',
        'frames_count',
        'processing_time',
        'frame_rate'
    )
    readonly_fields = (
        'id',
        'frames_count',
        'processing_time',
        'frame_rate'
    )


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    filter_horizontal = ('tags',)

    readonly_fields = (
        'id',
        'status',
        'created_at',
        'updated_at',
        'started_at',
        'finished_at',
        'progress',
        'info'
    )


@admin.register(Stat)
class StatAdmin(admin.ModelAdmin):
    fields = (
        'id',
        'name',
        'timestamp',
        'updated_at',
        'value',
        'resolution'
    )

    readonly_fields = (
        'id',
        'name',
        'timestamp',
        'updated_at',
        'value',
        'resolution'
    )


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):

    fields = (
        'id',
        'category',
        'dtype',
        'title',
        'message',
        'timestamp',
        'resource',
        'seen',
    )

    readonly_fields = (
        'id',
        'category',
        'dtype',
        'title',
        'message',
        'timestamp',
        'resource',
        'seen',
    )


@admin.register(Worker)
class WorkerAdmin(admin.ModelAdmin):
    pass
