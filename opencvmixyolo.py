import cv2
import numpy as np

# Orijinal görüntüyü yüklüyoruz.
original_image = cv2.imread("dene2.png")

# 1. Gri Tonlama uyguluyoruz
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# 2. Gaussian Blur uyguluyoruz
blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 1)

# 3. Canny Kenar Algılama uyguluyoruz
edges = cv2.Canny(blurred_image, 50, 150)

# 4. Morfolojik İşlemler
kernel = np.ones((2, 2), np.uint8)
dilated_image = cv2.dilate(edges, kernel, iterations=1)
eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

# 5. Maskeleme ve Görüntü Ağırlıklandırma yapıyoruz
masked_image = cv2.bitwise_and(original_image, original_image, mask=eroded_image)
filtered_image = cv2.addWeighted(original_image, 0.9, masked_image, 0.1, 0)
filtered_image = cv2.addWeighted(original_image, 0.9, masked_image, 0.1, 0)


# Görüntünün son filtrelenmiş  halini ekranda gösteriyoruz
cv2.imshow("Filtered Image", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# YOLO ağı ve sınıf isimlerini yüklüyoruz
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

image = filtered_image
original_image = image.copy()  
height, width, channels = image.shape

# Park yerlerini seçmek için global boş bir liste oluşturuyoruz
park_places = []

# Park yerlerini manuel olarak seçme fonksiyonu
def select_park_places(event, x, y, flags, param):
    global park_places, image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Seçilen koordinatları park_place listesine ekliyoruz otoparktaki tüm park yerlerinin bir defaya mahsus konumunu belirlemiş olduk.
        park_places.append([x, y])
        
        # Seçilen yerleri yeşil bir dikdörtgen içine alıyoruz.
        if len(park_places) % 2 == 0:
            start_x, start_y = park_places[-2]
            end_x, end_y = park_places[-1]
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)  # Yeşil dikdörtgen
            cv2.imshow("Otopark Seçimi", image)

cv2.imshow("Otopark Seçimi", image)
cv2.setMouseCallback("Otopark Seçimi", select_park_places)

while True:
    cv2.imshow("Otopark Seçimi", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # 'q' tuşuna basıldığında seçim bitirilecek diğer aşamaya geçilecek şekilde ayarlıyoruz.
        break

cv2.destroyAllWindows()

# Seçilen park yerlerininin konumlarını konsola yazdırıyoruz.
print("Seçilen Park Yerleri:", park_places)

# Görüntüyü blob’a çeviriyoruz (Yolo ya uygun hale gelmesi için)
blob = cv2.dnn.blobFromImage(original_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)


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


cv2.imshow("Otopark Durumu", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
