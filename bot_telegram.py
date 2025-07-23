import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
path_txt = 'data.txt'

loader = TextLoader(path_txt,encoding='utf-8')
txt = loader.load()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
indexs = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
chat_bot = ChatOpenAI(model='gpt-3.5-turbo', api_key=OPENAI_API_KEY, temperature=0)

chains = ConversationalRetrievalChain.from_llm(
    llm=chat_bot,
    retriever=indexs.vectorstore.as_retriever(search_kwargs={'k': 1})
)

user_history = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    usr_id = update.effective_chat.id
    message = update.message.text
    if usr_id not in user_history:
        user_history[usr_id] = []
    result = chains({
        'question': message,
        'chat_history': user_history[usr_id]
    })
    user_history[usr_id].append((message, result['answer']))
    await update.message.reply_text(result['answer'])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('مرحبا كيف اساعدك سيد محمد؟')

app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("✅ البوت يعمل...")
app.run_polling()
