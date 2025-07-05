import logging
import os
import asyncio
from quart import Quart, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, ContextTypes, MessageHandler, CommandHandler, CallbackQueryHandler, filters
from wakili_engine import WakiliAI, Config

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HELP_MESSAGE = """*Wakili Wangu Bot Help* ⚖️ ... (your help text here) ..."""
NON_TEXT_MESSAGE = "I'm sorry, I only understand text-based legal questions."


# --- Core Bot Handlers ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_MESSAGE, parse_mode='Markdown')


async def non_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(NON_TEXT_MESSAGE)


async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    logger.info(f"Feedback from {query.from_user.id}: {query.data}")
    await query.edit_message_text(text=f"{query.message.text}\n\n--- \n*🙏 Thank you for your feedback!*")


async def process_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Processing question from chat_id {chat_id}")
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        wakili_instance = context.application.wakili_ai_instance
        # Run the synchronous, CPU/network-bound AI call in a separate thread
        final_response = await asyncio.to_thread(wakili_instance.get_response, user_question)

        keyboard = [[
            InlineKeyboardButton("👍 Helpful", callback_data="feedback_helpful"),
            InlineKeyboardButton("👎 Not Helpful", callback_data="feedback_not_helpful"),
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(final_response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error handling question from {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("I'm sorry, an internal error occurred.")


# --- App Initialization & Server ---
app = Quart(__name__)
application = None


@app.before_serving
async def startup():
    global application
    try:
        config = Config()
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        WEBHOOK_URL = os.getenv('WEBHOOK_URL')

        if not all([TELEGRAM_TOKEN, WEBHOOK_URL]):
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or WEBHOOK_URL environment variables.")

        # Initialize our AI engine once
        wakili_instance = WakiliAI(config)

        # Build the Telegram application
        ptb_app = Application.builder().token(TELEGRAM_TOKEN).build()
        ptb_app.wakili_ai_instance = wakili_instance  # Attach the engine to the app context

        ptb_app.add_handler(CommandHandler("help", help_command))
        ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_text_message))
        ptb_app.add_handler(MessageHandler(filters.ALL & ~filters.TEXT & ~filters.COMMAND, non_text_handler))
        ptb_app.add_handler(CallbackQueryHandler(feedback_handler))

        await ptb_app.initialize()
        await ptb_app.start()
        await ptb_app.bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}", allowed_updates=Update.ALL_TYPES)

        application = ptb_app
        logger.info("Wakili Wangu Telegram Bot is now live.")
    except Exception as e:
        logger.critical(f"FATAL: Application failed to initialize. Error: {e}", exc_info=True)


@app.route("/health")
async def health_check():
    return "OK", 200


@app.route(f"/{os.getenv('TELEGRAM_BOT_TOKEN', 'default')}", methods=["POST"])
async def webhook_handler():
    if application:
        try:
            update = Update.de_json(await request.get_json(), application.bot)
            asyncio.create_task(application.process_update(update))
        except Exception as e:
            logger.error(f"Error processing webhook: {e}", exc_info=True)
        return "ok"
    return "Internal Server Error", 500