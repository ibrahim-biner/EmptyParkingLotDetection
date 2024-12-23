from django import forms
from detection.models import ParkingPhoto

class PhotoUploadForm(forms.ModelForm):
    class Meta:
        model = ParkingPhoto
        fields = ['photo']
