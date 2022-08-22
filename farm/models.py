from django.db import models

class PHL(models.Model):

	ph_level = models.CharField(max_length=200, null=True, blank=True)

	def __str__(self):

		return f"{self.ph_level}"

# Create your models here.
