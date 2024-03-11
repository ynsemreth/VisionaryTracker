# VISIONARY TRACKER

Bu proje, video akışlarında veya bir web kamerasından gerçek zamanlı nesne algılama ve izleme için YOLOv9 algoritmasından yararlanır. Algılama sınırlayıcı kutuları çizmek, nesne hareketlerini izlemek, görselleştirmeyi izlemek için ısı haritaları oluşturmak ve izleme sonuçlarını kaydetmek için işlevler içerir.

## Eğitim 

```sh
python train.py --workers 2 --device 'cpu' --batch 4 --data C:\Users\cypoi\Masaüstü\VisionaryTracker\data\carandperson\data.yaml --img 640 --cfg C:\Users\cypoi\Masaüstü\VisionaryTracker\models\detect\gelan-c.yaml --weights 'C:\Users\cypoi\Masaüstü\VisionaryTracker\gelan-c.pt' --name kisi --hyp C:\Users\cypoi\Masaüstü\VisionaryTracker\data\hyps\hyp.scratch-high.yaml --min-items 0 --epochs 10 --close-mosaic 15
```

## Özellikler

- **Nesne Algılama**: Videonun her karesindeki nesneleri tespit etmek için YOLOv9'u kullanır.
- **Nesne İzleme**: Nesneleri kareler boyunca izler ve yollarını görselleştirir.
- **Isı Haritası Oluşturma**: Nesnelerin izlenen yollarına dayalı bir ısı haritası oluşturur.
- **Özelleştirilebilir Video Girişi**: Video dosyalarını veya canlı web kamerası akışlarını işlemeyi destekler.
- **Seçilen Nesnenin Özelliklerini Çıkarma (HOG algortimasıyla)** : Roi ile seçilen nesnenin hem fotoğrafını fram frame hemde hog özelliklerini .txt dosyasına kaydeder.
- **Sonuç Kaydetme**: Algılanan nesnelerin ve izleme bilgilerinin bir özetini kaydeder.

## Nasıl Çalışır

### Video Girişi İşleme
- Kod, `--video` bağımsız değişkeni aracılığıyla bir video dosyası yolunu kabul eder. Herhangi bir yol belirtilmezse, varsayılan olarak web kamerasını kullanır (`0`).

### Nesne Algılama ve İzleme
- YOLOv9 önceden eğitilmiş ağırlıklarla yüklenir ve `data.yaml` dosyasında tanımlanan nesneleri algılamak üzere yapılandırılır.
- Algılanan nesneler, hareket yolları görselleştirilerek kareler boyunca izlenir.

### Tek ROI Seçimi

Tek bir ROI seçmek için projeyi çalıştırın ve video akışının başlamasını bekleyin. Akış canlı olduğunda:

1. Tek ROI seçim moduna girmek için `s` tuşuna basın.
2. Fareyi izlemek istediğiniz alanın üzerine tıklayın ve sürükleyin.
3. ROI'yi sonlandırmak için fare düğmesini bırakın.
4. Sistem artık bu belirtilen bölge içindeki nesneleri tespit etmeye ve izlemeye odaklanacaktır.

Bu özellik, izlemeyi sahnedeki belirli bir nesneye veya alana izole etmek için özellikle kullanışlıdır ve izleme doğruluğunu ve verimliliğini artırır.

### Çoklu ROI Seçimi

Aynı anda birkaç alana dikkat gerektiren senaryolar için sistemimiz birden fazla ROI seçimine izin vermektedir:

1. Çoklu ROI seçimini başlatmak için `f` tuşuna basın.
2. Her bir ROI için, fareyi istenen alanın üzerine tıklayıp sürükleyin ve sonlandırmak için bırakın.
3. Bir ROI seçtikten sonra, ek ROI'ler seçmeye devam etmek için `f` tuşuna tekrar basın.
4. Seçim işlemini tamamlamak ve izlemeye başlamak için `g` tuşuna basın.

Seçilen her ROI ayrı ayrı izlenecek ve aynı karede birden fazla ilgi alanındaki nesnelerin algılanmasına ve izlenmesine olanak sağlayacaktır.

### Etkili ROI Seçimi için İpuçları

- İzleme performansını etkileyebilecek çakışmaları önlemek için her bir ROI'nin sınırlarının net olduğundan emin olun.
- Çok sayıda alanın aynı anda izlenmesi hesaplama yükünü artırabileceğinden ve performansı etkileyebileceğinden, çoklu ROI özelliğini akıllıca kullanın.
- ROI seçimi, seçim işlemi yeniden başlatılarak herhangi bir zamanda ayarlanabilir veya sıfırlanabilir.

Kullanıcılar ROI seçimini kullanarak nesne algılama ve izleme sürecini belirli ihtiyaçlara göre uyarlayabilir, hesaplama kaynaklarını ilgili alanlara odaklayabilir ve genel sistem verimliliğini artırabilir.


### Isı Haritası Görselleştirme
- Çerçevenin farklı alanlarındaki nesne hareketlerinin sıklığını görselleştirmek için bir ısı haritası oluşturulur ve gerçek zamanlı olarak güncellenir.
![Heatmap Visualization](https://github.com/ynsemreth/VisionaryTracker/blob/main/result/heatmap.jpg)

### Sonuçları Kaydetme
- Algılanan nesneler ve izleme bilgileri bir dosyaya kaydedilerek kare başına nesne sayılarının bir özeti ve ayrıntılı izleme verileri sağlanır.

## Uygulama Ayrıntıları

### Bağımlılıklar
- argparse, cv2 (OpenCV), numpy ve `object_detector` ve `utils.detections` gibi özel modüller.

### Anahtar Fonksiyonlar
- `calculate_overlap`: İzlemeye yardımcı olmak için Birlik Üzerindeki Kesişimi (IoU) hesaplar.
- `save_tracking_results`: İzleme bilgilerini bir metin dosyasına kaydeder.
- add_weighted_heat`: Nesne konumlarına göre ısı haritasını günceller.
- Klavye etkileşimleri (`q`, `s`, `f`, `d`) çıkmak, izlemeyi başlatmak, birden fazla ROI seçmek ve izleme modunu devre dışı bırakmak için.

## Kurulum ve Kullanım

1. Bağımlılıkları yükleyin: Python 3.x'in gerekli paketlerle (OpenCV, NumPy) birlikte yüklendiğinden emin olun.
2. YOLOv7 model ağırlıklarınızı ve yapılandırma dosyalarınızı proje dizinine yerleştirin.
3. Komut dosyasını istediğiniz video girişi ile çalıştırın veya web kamerası girişi için:
```sh
python main.py

python main.py --video ./videos/examples.mp4
```

4. İzleme süreciyle etkileşim kurmak için yürütme sırasında klavye komutlarını kullanın.

## Sonuç

Bu proje, hareket analizi için ısı haritası görselleştirmesi ile geliştirilmiş gerçek zamanlı nesne algılama ve izleme için YOLOv9'un gücünü sergiliyor. Gözetimden spor analitiğine kadar çeşitli uygulamalar için uyarlanabilir.

# ALGORITMALAR : 

## HOG (Histogram of Oriented Gradients)

### Amaç: 
HOG, görüntülerdeki şekil ve doku bilgilerini yakalamak için tasarlanmış bir özellik çıkartma yöntemidir.
    
### Çalışma Prensibi:
Görüntüdeki her bir pikselin gradyan yönünü ve büyüklüğünü hesaplar. Daha sonra, bu gradyanlar belirli bir pencere içerisindeki hücrelere ayrılır ve her bir hücre için gradyan yönlerinin histogramı oluşturulur. Bu histogramlar, görüntünün yerel gradyan yapısını özetleyen güçlü ve açıklayıcı özellikler üretir.

### Kullanım Alanları: 
Yaya tespiti, araç tanıma ve insan tanıma gibi görevlerde yaygın olarak kullanılır.

## SIFT (Scale-Invariant Feature Transform)

### Amaç: 
SIFT, görüntülerdeki anahtar noktaları bulmak ve bunların özelliklerini çıkarmak için kullanılır. Bu algoritma, ölçek ve dönüşüme karşı dayanıklı özellikler sağlar.
### Çalışma Prensibi:
Görüntü üzerinde ölçek uzayı aşamaları uygulanır, potansiyel ilgi noktaları tespit edilir, ve bu noktaların çevresindeki gradyan bilgileri kullanılarak her bir nokta için benzersiz bir tanımlayıcı (descriptor) oluşturulur.
### Kullanım Alanları: 
Nesne tanıma, panoramik görüntü birleştirme ve 3D modelleme gibi çeşitli alanlarda kullanılır.

## SURF (Speeded Up Robust Features)

### Amaç: 
SURF, SIFT'in hedeflediği sorunları çözmek için tasarlanmıştır ancak daha hızlı hesaplama ve benzer veya daha iyi performans sunar.
### Çalışma Prensibi: 
Hızlı bir şekilde ilgi noktalarını tespit etmek için Hessian matris tabanlı bir yaklaşım kullanır. Bu noktalar için özellik tanımlayıcıları, ilgi noktalarının çevresindeki basit, hızlı ve etkili bir şekilde hesaplanabilir şekilde üretilir.
### Kullanım Alanları: 
SIFT'e benzer şekilde, nesne tanıma, görüntü eşleştirme ve 3D rekonstrüksiyon gibi alanlarda kullanılır.
