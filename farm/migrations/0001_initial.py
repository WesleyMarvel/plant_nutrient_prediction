# Generated by Django 3.2.5 on 2021-09-17 16:22

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PHL',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ph_level', models.CharField(blank=True, max_length=200, null=True)),
            ],
        ),
    ]
