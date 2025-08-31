GPT-2 Fine-Tuning ve Basit Chatbot Örneği
Bu Python betiği, Hugging Face transformers kütüphanesini kullanarak önceden eğitilmiş bir GPT-2 modelini özel bir metin verisi üzerinde fine-tune (ince ayar) yapar ve ardından eğitilmiş modeli kullanarak basit bir sohbet botu oluşturur.

Gereksinimler
Bu projeyi çalıştırmak için gerekli kütüphaneleri aşağıdaki komut ile kurabilirsiniz:

Bash

pip install transformers torch datasets
Kullanım
Betiği yerel olarak kaydedin (fine_tune_chatbot.py gibi).

Terminalinizde aşağıdaki komutu çalıştırın:

Bash

python fine_tune_chatbot.py
Betiği çalıştırdığınızda, fine-tuning süreci otomatik olarak başlayacak ve tamamlandığında modeliniz ./fine_tuned_gpt2 klasörüne kaydedilecektir. Daha sonra etkileşimli bir sohbet botu oturumu başlayacaktır.

Nasıl Çalışır?
Kod, aşağıdaki adımları sırasıyla uygular:

Model ve Tokenizer Yükleme: Temel GPT-2 modeli ve tokenizer'ı Hugging Face Hub'dan indirilir.

Veri Hazırlama: Fine-tuning için kullanılacak özel metin verisi tanımlanır ve datasets kütüphanesi formatına dönüştürülür.

Fine-tuning: Trainer sınıfı kullanılarak model, yeni veri üzerinde eğitilir. Bu süreçte, modelin ağırlıkları yeni veriye göre güncellenir.

Modeli Kaydetme: Eğitim tamamlandıktan sonra, ince ayar yapılmış model ve tokenizer yerel diske kaydedilir.

Sohbet Botu: Kaydedilen model yeniden yüklenir ve kullanıcıdan gelen metin girişlerine göre metin üretimi yapar. Çıkmak için exit yazmanız yeterlidir.
