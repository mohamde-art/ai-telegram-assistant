# ai-telegram-assistant
# 🤖 Telegram AI Knowledge Bot

بوت تلغرام ذكي يعتمد على **LangChain** و **OpenAI GPT** للإجابة عن الأسئلة بناءً على قاعدة معرفة محلية (ملف نصي).  
يمكن استخدامه كـ **مساعد شخصي** أو **بوت دعم فني** للإجابة عن الأسئلة الشائعة.

---

## 🚀 المميزات
- محادثة ذكية مع المستخدمين عبر **تلغرام**.
- يعتمد على نموذج **GPT-3.5 Turbo** من OpenAI.
- إمكانية البحث عن الإجابات في **قاعدة معرفة نصية** (ملف `data.txt`).
- حفظ **سجل المحادثات** لكل مستخدم لإجابات سياقية أدق.
- تصميم بسيط وسهل التخصيص.

---

## 🛠️ المتطلبات

- Python 3.9+
- حساب في [OpenAI](https://platform.openai.com/account/api-keys) للحصول على **API Key**.
- إنشاء بوت في [BotFather](https://t.me/BotFather) للحصول على **Telegram Bot Token**.

---

## 📦 التثبيت

1. **استنساخ المشروع:**
```bash
git clone https://github.com/username/telegram-ai-knowledge-bot.git
cd telegram-ai-knowledge-bot
python -m venv venv
source venv/bin/activate   # (على Linux/Mac)
venv\Scripts\activate      # (على Windows)

pip install -r requirements.txt

OPENAI_API_KEY=your_openai_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
data.txt

ai-telegram-assistant
│
├── bot.py               # الكود الرئيسي للبوت
├── data.txt             # قاعدة المعرفة النصية
├── .env                 # بيانات المفاتيح السرية
├── requirements.txt     # المكتبات المطلوبة
└── README.md            # ملف التوثيق

هذا الملف يحتوي على المعلومات التي سيبحث فيها البوت للإجابة على أسئلة المستخدم.
يمكنك وضع أي نص تريده هنا.
