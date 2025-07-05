# telegram_bot.py

import logging
import os
import asyncio
from flask import Flask, request, Response
from telegram import Update, Bot
from telegram.ext import Application, ContextTypes, TypeHandler
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

    # We create the application instance here
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.wakili_ai_instance = wakili_instance

except Exception as e:
    logger.critical(f"FATAL: Application failed to initialize. Error: {e}", exc_info=True)
    wakili_instance = None
    application = None

# --- Web Server (Flask) Setup ---
app = Flask(__name__)


# --- Telegram Processing Logic ---
async def process_telegram_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The core logic to handle a text message."""
    user_question = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Received question from chat_id {chat_id}: \"{user_question}\"")

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        wakili_instance = context.application.wakili_ai_instance
        final_response = wakili_instance.get_wakili_response(user_question)
        await update.message.reply_text(final_response, parse_mode='Markdown')
        logger.info(f"Successfully sent response to chat_id {chat_id}.")
    except Exception as e:
        logger.error(f"Error handling question from {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("I'm sorry, an internal error occurred. Please try again.")


@app.route("/health")
def health_check():
    """Render's health check endpoint."""
    return "OK", 200


@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
async def webhook_handler():
    """Handles incoming updates by passing them to the application."""
    if not application:
        return "Internal Server Error", 500

    update_data = request.get_json()
    update = Update.de_json(update_data, application.bot)

    # This is a fire-and-forget task. Flask returns "ok" immediately,
    # and the telegram library processes the update in the background.
    asyncio.create_task(application.process_update(update))

    return "ok"


async def main():
    """Initializes the bot and sets the webhook."""
    if not application:
        logger.error("Application object not created. Cannot start.")
        return

    # Add a handler for text messages. We moved the logic to its own function.
    application.add_handler(TypeHandler(Update, process_telegram_update))

    # --- THE FIX IS HERE ---
    # We must initialize the application before using it
    logger.info("Initializing application...")
    await application.initialize()
    logger.info("Starting application...")
    await application.start()  # Starts background tasks
    # --- END OF FIX ---

    logger.info(f"Setting webhook to {WEBHOOK_URL}/{TELEGRAM_TOKEN}")
    await application.bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}", allowed_updates=Update.ALL_TYPES)
    logger.info("Webhook set successfully.")


if __name__ == "__main__":
    if wakili_instance and application:
        # Run the async main function to initialize and set the webhook
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())

        # Start the Flask web server
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port)
    else:
        logger.critical("Could not start Flask server because initialization failed.")