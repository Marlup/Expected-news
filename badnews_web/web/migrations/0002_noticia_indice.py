# Generated by Django 4.2.1 on 2023-05-14 21:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='noticia',
            name='indice',
            field=models.IntegerField(default=0),
        ),
    ]
