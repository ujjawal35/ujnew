from django.conf import settings
from django.db import models
from django.utils import timezone


class BlogEntry(models.Model):
    author = models.CharField(max_length=200)
    title = models.CharField(max_length=200)
    text = models.TextField()
    created_date = models.DateTimeField(default=timezone.now)
    published_date = models.DateTimeField(blank=True, null=True)

    def publishOnSite(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.title