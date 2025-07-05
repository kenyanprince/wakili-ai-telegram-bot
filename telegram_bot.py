# telegram_bot.py

import logging
import os
import asyncio
from quart import Quart, request
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from wakili_engine import WakiliAI, Config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
wakili_instance = None
application = None
TELEGRAM_TOKEN = None
WEBHOOK_URL = None


# --- Message Handler Function ---
async def process_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The core logic to handle a text message."""
    user_question = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Received question from chat_id {chat_id}: \"{user_question}\"")

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        wakili_instance = context.application.wakili_ai_instance

        logger.info("Calling WakiliAI engine in a background thread...")
        final_response = await asyncio.to_thread(
            wakili_instance.get_wakili_response, user_question
        )
        logger.info("WakiliAI engine has returned a response.")

        await update.message.reply_text(final_response, parse_mode='Markdown')
        logger.info(f"Successfully sent final response to chat_id {chat_id}.")

    except Exception as e:
        logger.error(f"Error handling question from {chat_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "I'm sorry, an internal error occurred while processing your request. Please try again.")


# --- Initialization Function ---
async def initialize_app():
    """Initialize the Telegram bot application."""
    global wakili_instance, application, TELEGRAM_TOKEN, WEBHOOK_URL

    try:
        config = Config()
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        WEBHOOK_URL = os.getenv('WEBHOOK_URL')

        if not all([TELEGRAM_TOKEN, WEBHOOK_URL]):
            raise ValueError("Missing critical environment variables: TELEGRAM_BOT_TOKEN, WEBHOOK_URL")

        wakili_instance = WakiliAI(config)

        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.wakili_ai_instance = wakili_instance

        # Add the message handler
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_text_message))

        # Initialize and start the application
        await application.initialize()
        await application.start()
        await application.bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}", allowed_updates=Update.ALL_TYPES)

        logger.info("Bot setup complete. Webhook is set.")
        return True

    except Exception as e:
        logger.critical(f"FATAL: Application failed to initialize. Error: {e}", exc_info=True)
        return False


# --- Quart App (ASGI compatible) ---
app = Quart(__name__)


@app.before_serving
async def startup():
    """Initialize the bot when the app starts."""
    success = await initialize_app()
    if not success:
        logger.critical("Failed to initialize bot. App will not function properly.")


@app.route("/health")
async def health_check():
    return "OK", 200


@app.route(f"/{os.getenv('TELEGRAM_BOT_TOKEN', 'default')}", methods=["POST"])
async def webhook_handler():
    global application

    if not application:
        return "Internal Server Error", 500

    try:
        json_data = await request.get_json()
        update = Update.de_json(json_data, application.bot)
        asyncio.create_task(application.process_update(update))
        return "ok"
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        return "Error", 500


@app.route("/")
async def root():
    return "Wakili AI Telegram Bot is running!"


if __name__ == "__main__":
    # This won't run when using Hypercorn, but useful for local testing
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))