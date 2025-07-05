# telegram_bot.py

import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from wakili_engine import WakiliAI, Config  # Import your engine

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Bot Welcome Message ---
WELCOME_MESSAGE = """
Karibu! Welcome to *Wakili Wangu Bot*  правосуддя! ⚖️

I am your AI legal assistant for Kenyan law. I can help you understand legal concepts by searching through statutes and case law.

*How to use me:*
Just ask me a legal question in plain English. For example:
- "What are my rights if I am arrested?"
- "How do I register a company in Kenya?"

Please wait a moment after asking, as I need to think and research.

---
*_Disclaimer:_* _I am an AI bot and not a qualified advocate. My responses are for informational purposes only and do not constitute legal advice. Always consult with a registered legal professional for your specific situation._
"""


# --- Telegram Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a welcome message when the /start command is issued."""
    await update.message.reply_text(WELCOME_MESSAGE, parse_mode='Markdown')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles regular user messages, treating them as legal questions."""
    user_question = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Received question from chat_id {chat_id}: \"{user_question}\"")

    # Let the user know the bot is working on it
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        # Get the WakiliAI instance from the application context
        wakili_instance = context.application.wakili_ai_instance

        # Get the legal response
        final_response = wakili_instance.get_wakili_response(user_question)

        await update.message.reply_text(final_response, parse_mode='Markdown')
        logger.info(f"Successfully sent response to chat_id {chat_id}.")

    except Exception as e:
        logger.error(f"Error handling question from {chat_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "I'm sorry, I encountered an internal error. Please try asking your question again in a few moments.")


def main():
    """Starts the Telegram bot."""
    # --- Load Configuration and Initialize WakiliAI ---
    try:
        config = Config()
        # IMPORTANT: Load the NEW Telegram token from environment variables
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        if not TELEGRAM_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set!")

        wakili_instance = WakiliAI(config)
        logger.info("WakiliAI engine initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Could not initialize WakiliAI engine. {e}", exc_info=True)
        return  # Exit if the core engine can't start

    # --- Set up the Telegram Application ---
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Store the single instance of WakiliAI in the application context
    # This is efficient as it avoids re-initializing the engine for every message
    application.wakili_ai_instance = wakili_instance

    # --- Register Handlers ---
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # --- Start the Bot ---
    logger.info("Starting Telegram bot polling...")
    application.run_polling()


if __name__ == "__main__":
    main()