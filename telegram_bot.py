# telegram_bot.py

import logging
import os
import asyncio
from flask import Flask, request, Response
from telegram import Update
from telegram.ext import Application, ContextTypes
from wakili_engine import WakiliAI, Config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Initialization ---
try:
    config = Config()
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    WEBHOOK_URL = os.getenv('WEBHOOK_URL')  # The public URL of your Render service

    if not all([TELEGRAM_TOKEN, WEBHOOK_URL]):
        raise ValueError("Missing critical environment variables: TELEGRAM_BOT_TOKEN, WEBHOOK_URL")

    # Initialize WakiliAI engine once
    wakili_instance = WakiliAI(config)

    # Set up the Telegram Application
    # We don't need command handlers here as all updates come through the webhook
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Store the single instance of WakiliAI for access within handlers
    application.wakili_ai_instance = wakili_instance

except Exception as e:
    logger.critical(f"FATAL: Application failed to initialize. Error: {e}", exc_info=True)
    # If initialization fails, we should not proceed.
    # In a real app, this would cause the container to fail and restart.
    wakili_instance = None
    application = None

# --- Web Server (Flask) Setup ---
app = Flask(__name__)


@app.route("/health")
def health_check():
    """Render's health check endpoint."""
    return "OK", 200


@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
async def webhook_handler():
    """Handles incoming updates from Telegram."""
    if not application:
        logger.error("Application not initialized, cannot handle webhook.")
        return "Internal Server Error", 500

    update_data = request.get_json()
    update = Update.de_json(update_data, application.bot)

    # Process the update in the background
    asyncio.create_task(application.process_update(update))

    return "ok"


async def main():
    """Sets the Telegram webhook."""
    if not application:
        logger.error("Application not initialized, cannot set webhook.")
        return

    logger.info(f"Setting webhook to {WEBHOOK_URL}/{TELEGRAM_TOKEN}")
    await application.bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}", allowed_updates=Update.ALL_TYPES)
    logger.info("Webhook set successfully.")


if __name__ == "__main__":
    # This block now starts the web server, not the bot polling
    if wakili_instance and application:
        # Run the async main function to set the webhook
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())

        # Start the Flask web server
        # Render will pass the PORT environment variable automatically
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port)
    else:
        logger.critical("Could not start Flask server because initialization failed.")