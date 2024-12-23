from django.db import models

class ParkingPhoto(models.Model):
    photo = models.ImageField(upload_to='original_photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class FilteredPhoto(models.Model):
    parking_photo = models.ForeignKey(ParkingPhoto, related_name='filtered_photos', on_delete=models.CASCADE)
    filter_step = models.IntegerField()  # 1, 2, 3, 4, 5 adımlarını belirtir
    filtered_image = models.ImageField(upload_to='filtered_photos/')
    processed_at = models.DateTimeField(auto_now_add=True)

class FinalResult(models.Model):
    parking_photo = models.ForeignKey(ParkingPhoto, related_name='final_result', on_delete=models.CASCADE)
    result_image = models.ImageField(upload_to='result_photos/')
    processed_at = models.DateTimeField(auto_now_add=True)
