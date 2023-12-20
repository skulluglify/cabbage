import uuid

from django.contrib.auth.models import User
from django.db import models


class ProfileModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    nickname = models.CharField(max_length=50, blank=True)
    gender = models.CharField(max_length=20, blank=True)
    website = models.URLField(blank=True)
    picture = models.ImageField(blank=True)
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    def mp_update(self):
        self.updated_at = models.DateTimeField(auto_now=True)
        self.save()

    def mp_delete(self):
        self.deleted_at = models.DateTimeField(auto_now=True)
        self.save()


class RecordModel(models.Model):
    ph = models.FloatField(default=0.0)
    ec = models.FloatField(default=0.0)
    air_temp = models.FloatField(default=0.0)
    humidity = models.FloatField(default=0.0)
    water_flow = models.BooleanField(default=False)
    lighting = models.BooleanField(default=False)
    full_water_tank = models.BooleanField(default=False)
    acid_actuator = models.BooleanField(default=False)
    alkaline_actuator = models.BooleanField(default=False)
    nutrient_actuator = models.BooleanField(default=False)
    fans_rpm = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now=True)


class CommandModel(models.Model):
    lighting = models.BooleanField(default=False)
    acid_actuator = models.BooleanField(default=False)
    alkaline_actuator = models.BooleanField(default=False)
    nutrient_actuator = models.BooleanField(default=False)
    fans_rpm = models.IntegerField(default=0)
    accepted = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now=True)


class ForecastModel(models.Model):
    image = models.URLField()
    result = models.JSONField()
    created_at = models.DateTimeField(auto_now=True)


class PlantModel(models.Model):
    user_id = models.ForeignKey(ProfileModel, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    planted = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now=True)
