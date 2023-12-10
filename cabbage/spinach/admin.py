from django.contrib import admin
from .models import ProfileModel, PlantModel, ForecastModel, CommandModel, RecordModel


# admin model register!

class ProfileAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'user__username',
        'user__email',
        'nickname',
        'user__is_staff',
        'user__is_superuser',
        'created_at',
        'updated_at',
    ]
    search_fields = [
        'user__username',
        'user__email',
        'nickname',
    ]

    @admin.display(ordering='user__username', description='Username')
    def user__username(self, obj):
        return obj.user.username

    @admin.display(ordering='user__email', description='Email')
    def user__email(self, obj):
        return obj.user.email

    @admin.display(ordering='user__is_staff', description='Staff')
    def user__is_staff(self, obj):
        return obj.user.is_staff

    @admin.display(ordering='user__is_superuser', description='Superuser')
    def user__is_superuser(self, obj):
        return obj.user.is_superuser


class RecordAdmin(admin.ModelAdmin):
    list_display = [
        'ph',
        'ec',
        'air_temp',
        'humidity',
        'water_flow',
        'lighting',
        'full_water_tank',
        'acid_actuator',
        'alkaline_actuator',
        'nutrient_actuator',
        'fans_rpm',
        'created_at',
    ]
    search_fields = []


class CommandAdmin(admin.ModelAdmin):
    list_display = [
        'lighting',
        'acid_actuator',
        'alkaline_actuator',
        'nutrient_actuator',
        'fans_rpm',
        'accepted',
        'created_at',
    ]
    search_fields = []


class ForecastAdmin(admin.ModelAdmin):
    list_display = ['image', 'result', 'created_at']
    search_fields = ['image']


class PlantAdmin(admin.ModelAdmin):
    list_display = ['name', 'planted', 'created_at']
    search_fields = ['name']


# admin sites register!

admin.site.register(ProfileModel, ProfileAdmin)
admin.site.register(RecordModel, RecordAdmin)
admin.site.register(CommandModel, CommandAdmin)
admin.site.register(ForecastModel, ForecastAdmin)
admin.site.register(PlantModel, PlantAdmin)
