from django.shortcuts import render, redirect
from detection.forms import PhotoUploadForm
from detection.models import ParkingPhoto, FilteredPhoto, FinalResult
import cv2
import os
import numpy as np
from django.shortcuts import render
import os
from django.conf import settings
from detection.models import FilteredPhoto

def home(request):
    return render(request, 'home.html')  # home.html şablonunu render etmek

def save_final_result(parking_photo, final_image):
    # Göreceli dosya yolunu belirle
    relative_path = os.path.join('result_photos', f"final_result_{parking_photo.id}.jpg")
    # Medya dizinindeki tam dosya yolunu oluştur
    absolute_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    
    # Hedef dizini oluştur (eğer yoksa)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    
    # Görüntüyü tam dosya yoluna kaydet
    if not cv2.imwrite(absolute_path, final_image):
        raise ValueError(f"Final image could not be saved at {absolute_path}.")
    
    # Veritabanına göreceli dosya yolunu kaydet
    return FinalResult.objects.create(
        parking_photo=parking_photo,
        result_image=relative_path  # Sadece göreceli yol kaydediliyor
    )

def save_filtered_image(image, parking_photo, step):
    # Dosya adını ve yolu oluştur
    filename = f"filtered_step_{step}.jpg"
    relative_path = os.path.join('filtered_photos', filename)
    absolute_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    
    # Dizin kontrolü ve oluşturma
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    
    # Görüntüyü kaydet
    if not cv2.imwrite(absolute_path, image):
        raise ValueError(f"Filtered image could not be saved at {absolute_path}.")
    
    # Veritabanına kaydet
    return FilteredPhoto.objects.create(
        parking_photo=parking_photo,
        filter_step=step,
        filtered_image=relative_path
    )

def upload_photo(request):
    if request.method == 'POST' and request.FILES['photo']:
        form = PhotoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Fotoğrafı kaydet
            parking_photo = form.save()

            # OpenCV işlemleri burada yapılacak
            original_image = cv2.imread(parking_photo.photo.path)
            filtered_images = process_image(original_image)

            # Filtrelenmiş fotoğrafları kaydet
            for i, filtered_image in enumerate(filtered_images):
                filtered_image_path = save_filtered_image(filtered_image, parking_photo, i+1)

            # Yolo modeline gönderme
            final_image = apply_yolo(original_image)
            if final_image is None:
                raise ValueError("Final image could not be generated.")

            # Sonuç fotoğrafını kaydet
            save_final_result(parking_photo, final_image)

            return redirect('show_result', photo_id=parking_photo.id)
    else:
        form = PhotoUploadForm()

    return render(request, 'detection/upload_photo.html', {'form': form})

import cv2
import numpy as np

def process_image(image):
    # OpenCV ile 5 adımlı filtreleme
    filtered_images = []

    # 1. Gri Tonlama uyguluyoruz
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_images.append(gray_image)  # İlk adımı ekleyelim

    # 2. Gaussian Blur uyguluyoruz
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 1)
    filtered_images.append(blurred_image)  # İkinci adımı ekleyelim

    # 3. Canny Kenar Algılama uyguluyoruz
    edges = cv2.Canny(blurred_image, 50, 150)
    filtered_images.append(edges)  # Üçüncü adımı ekleyelim

    # 4. Morfolojik İşlemler
    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(edges, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
    filtered_images.append(eroded_image)  # Dördüncü adımı ekleyelim

    # 5. Maskeleme ve Görüntü Ağırlıklandırma yapıyoruz
    masked_image = cv2.bitwise_and(image, image, mask=eroded_image)
    filtered_image = cv2.addWeighted(image, 0.9, masked_image, 0.1, 0)
    #filtered_images.append(filtered_image)  # Beşinci adımı ekleyelim

    return filtered_images



def apply_yolo(image):

    net = cv2.dnn.readNet("detection/yolov4.weights", "detection/yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    with open("detection/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    image = image
    original_image = image.copy()  
    height, width, channels = image.shape

    # Park yerlerini seçmek için global boş bir liste oluşturuyoruz
    park_places = [[14, 78], [61, 157], [108, 85], [137, 156], [202, 79], [230, 149], [290, 91], [326, 156], [380, 84], [425, 151], [471, 93], [516, 151], [561, 96], [601, 153], [643, 100], [701, 166], [746, 101], [775, 151], [828, 99], [865, 157], [916, 100], [960, 157], [1001, 95], [1044, 154], [63, 204], [107, 280], [172, 200], [206, 273], [270, 204], [314, 278], [361, 199], [410, 279], [459, 198], [514, 274], [566, 208], [625, 292], [665, 209], [709, 285], [759, 204], [813, 295], [860, 208], [903, 288], [958, 215], [999, 297]]

    blob = cv2.dnn.blobFromImage(original_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    #################################################################################################################
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Yolo çıktıları için gerekli listeleri hazırlıyoruz.
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_id == 2:  # "car" sınıfı COCO içerisinde id==2 olanlar olduğu için sadece araba tanımaya ayarlıyoruz
                # Tanınan arabaların içinde bulundukları kutuların kordinatlarını alıyoruz
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # openCV ye ait Non-maximum suppression fonksiyonunu uygulayarak fazla kutuları temizliyoruz
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Arabaları tespit etttikten sonra dolu ve boş konumları tutacağımız listeleri oluşturuyoruz.
    dolu = []  
    bos = []   

    # Park yerlerini kontrolediyoruz ve dolu yerlerde kesişme varmı ona bakıyoruz kesişme var ise o alanı dolu olarak alıyoruz
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            if label == "car":
                car_rect = (x, y, w, h)  

                for i in range(0, len(park_places), 2):
                    start_x, start_y = park_places[i]
                    end_x, end_y = park_places[i + 1]

                    
                    park_rect = (start_x, start_y, end_x - start_x, end_y - start_y)

                    # Kesişim kontrolünün yapıldığı if bloğu
                    if (car_rect[0] < park_rect[0] + park_rect[2] and
                        car_rect[0] + car_rect[2] > park_rect[0] and
                        car_rect[1] < park_rect[1] + park_rect[3] and
                        car_rect[1] + car_rect[3] > park_rect[1]):
                        # Kesişim varsa, dolu park yeri olarak dolu aldı listeye ekliyoruz o konumu
                        dolu.append(park_rect)
                        break

    # Sonuç görüntüsününün oluşturlulma aşaması
    green_count = 0  # Yeşil kutuların sayısının tutulacı değişken
    total_count = len(park_places) // 2  # Tüm kutuların sayısının tutulacağı değişken

    for i in range(0, len(park_places), 2):
        start_x, start_y = park_places[i]
        end_x, end_y = park_places[i + 1]
        park_rect = (start_x, start_y, end_x - start_x, end_y - start_y)

        if park_rect not in dolu:
            cv2.rectangle(original_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)  
            green_count += 1
        else:
            pass  

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            if label == "car":
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  

    #Sonuç olarak kaç boş alan ve toplamda kaç park alanı olduğunu ekrana yazdırıyoruz.
    cv2.putText(original_image, f"Bos Alan Sayisi: {green_count}/{total_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    ####################################################################################################################
    final_image = original_image # YOLO sonucu burada işlenecek
    return final_image




def show_result(request, photo_id):
    parking_photo = ParkingPhoto.objects.get(id=photo_id)
    filtered_photos = parking_photo.filtered_photos.all()
    final_result = parking_photo.final_result.first()
    
    return render(request, 'detection/show_result.html', {
        'filtered_photos': filtered_photos,
        'final_result': final_result,
    })
