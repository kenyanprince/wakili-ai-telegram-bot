# telegram_bot.py

import logging
import os
import asyncio
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from wakili_engine import WakiliAI, Config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Initialization ---
try:
    config = Config()
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    WEBHOOK_URL = os.getenv('WEBHOOK_URL')

    if not all([TELEGRAM_TOKEN, WEBHOOK_URL]):
        raise ValueError("Missing critical environment variables: TELEGRAM_BOT_TOKEN, WEBHOOK_URL")

    wakili_instance = WakiliAI(config)

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.wakili_ai_instance = wakili_instance

except Exception as e:
    logger.critical(f"FATAL: Application failed to initialize. Error: {e}", exc_info=True)
    wakili_instance = None
    application = None

# --- Web Server (Flask) Setup ---
app = Flask(__name__)


# --- Telegram Processing Logic ---
async def process_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The core logic to handle a text message."""
    user_question = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Received question from chat_id {chat_id}: \"{user_question}\"")

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        wakili_instance = context.application.wakili_ai_instance

        # --- THE FIX IS HERE ---
        # Run the blocking function in a separate thread to avoid freezing the event loop
        logger.info("Calling WakiliAI engine in a background thread...")
        final_response = await asyncio.to_thread(
            wakili_instance.get_wakili_response, user_question
        )
        logger.info("WakiliAI engine has returned a response.")
        # --- END OF FIX ---

        await update.message.reply_text(final_response, parse_mode='Markdown')
        logger.info(f"Successfully sent final response to chat_id {chat_id}.")

    except Exception as e:
        logger.error(f"Error handling question from {chat_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "I'm sorry, an internal error occurred while processing your request. Please try again.")


@app.route("/health")
def health_check():
    return "OK", 200


@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
async def webhook_handler():
    if not application:
        return "Internal Server Error", 500

    update = Update.de_json(request.get_json(), application.bot)
    asyncio.create_task(application.process_update(update))

    return "ok"


async def main():
    if not application:
        return

    # More specific handler for text messages only
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_text_message))

    await application.initialize()
    await application.start()

    await application.bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}", allowed_updates=Update.ALL_TYPES)
    logger.info("Webhook setup complete. Bot is ready.")


if __name__ == "__main__":
    if wakili_instance and application:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())

        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port)
    else:
        logger.critical("Could not start Flask server because initialization failed.")