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

# --- App Initialization ---
# This part now runs when Hypercorn imports the file
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

    # Run the async setup ONCE when the app starts
    asyncio.run(application.initialize())
    asyncio.run(application.start())
    asyncio.run(application.bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}", allowed_updates=Update.ALL_TYPES))

    logger.info("Bot setup complete. Webhook is set.")

except Exception as e:
    logger.critical(f"FATAL: Application failed to initialize. Error: {e}", exc_info=True)
    wakili_instance = None
    application = None

# --- Web Server (Flask) and Route Handlers ---
app = Flask(__name__)


# This function's logic is now inside the try/except block above
# and is no longer needed
# async def main():
#     pass

# This is the main processing function
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

# THE FIX IS HERE: We no longer have an __main__ block that calls app.run()
# Hypercorn is now our entry point.