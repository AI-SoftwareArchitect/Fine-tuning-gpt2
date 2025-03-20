from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

# 1️⃣ Model ve Tokenizer yükleme
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Padding hatasını gidermek için PAD token ayarlama
tokenizer.pad_token = tokenizer.eos_token

# 2️⃣ Yeni veriyi belirleme
new_data = "Yapay zeka, günümüz dünyasında devrim niteliğinde ilerlemeler sağlıyor. Büyük dil modelleri, metin üretimi, çeviri ve anlam analizi gibi birçok alanda insanlara yardımcı oluyor. Fine-tuning süreci sayesinde, önceden eğitilmiş bir modeli belirli bir göreve daha uygun hale getirebiliriz."

# 3️⃣ Veriyi Dataset formatına çevirme
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

dataset = Dataset.from_dict({"text": [new_data]})
dataset = dataset.map(tokenize_function, batched=True)

# 4️⃣ Fine-tuning için ayarlar
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Masked Language Modeling sadece BERT için gerekli
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 5️⃣ Modeli eğitme
print("🚀 Fine-tuning başlıyor...")
trainer.train()
print("✅ Fine-tuning tamamlandı!")

# 6️⃣ Fine-tuned modeli kaydetme
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
print("📁 Model kaydedildi: ./fine_tuned_gpt2")

# 7️⃣ Fine-tuned modeli yükleme (chatbot için)
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 8️⃣ Model ile konuşma fonksiyonu
def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 9️⃣ Kullanıcı ile sohbet başlatma
print("\n🤖 GPT-2 Chatbot aktif! Çıkmak için 'exit' yaz.\n")

while True:
    prompt = input("📝 Soru sor veya bir cümle başlat: ")
    if prompt.lower() == "exit":
        print("👋 Görüşmek üzere!")
        break
    response = generate_response(prompt)
    print("\n🧠 Modelin cevabı:", response, "\n")
