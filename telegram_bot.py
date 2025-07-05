# telegram_bot.py

import logging
import os
import asyncio
from quart import Quart, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
)
from wakili_engine import WakiliAI, Config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
wakili_instance = None
application = None
TELEGRAM_TOKEN = None
WEBHOOK_URL = None

# --- Constants for Reusable Messages ---
HELP_MESSAGE = """
*Wakili Wangu Bot Help* ⚖️

I am an AI assistant designed to help you understand Kenyan law.

*How to Use Me:*
Simply ask a legal question in plain English. For example:
- "What are my rights if I am arrested?"
- "How do I register a company in Kenya?"
- "What happens in a hit and run case?"

I will search through Kenyan statutes and case law to give you a detailed, yet easy-to-understand explanation.

---
*_Disclaimer:_* _I am an AI, not a lawyer. My answers are for informational purposes only. Always consult a qualified advocate for legal advice._
"""

NON_TEXT_MESSAGE = "I'm sorry, I can only understand legal questions written in text. Please type your question for me."


# --- Core Bot Functions ---

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /help command."""
    await update.message.reply_text(HELP_MESSAGE, parse_mode='Markdown')


async def non_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for any message that is not text (e.g., photos, stickers)."""
    await update.message.reply_text(NON_TEXT_MESSAGE)


async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles button presses for feedback."""
    query = update.callback_query
    await query.answer()  # Important to answer the query first

    # The data will be 'feedback_helpful' or 'feedback_not_helpful'
    feedback = query.data
    user_id = query.from_user.id

    # For now, we just log the feedback. Later, this could be saved to a database.
    logger.info(f"Received feedback from user {user_id}: {feedback}")

    # Edit the original message to show feedback was received
    await query.edit_message_text(text=f"{query.message.text}\n\n--- \n*🙏 Thank you for your feedback!*")


async def process_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The core logic to handle a text message."""
    user_question = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Received question from chat_id {chat_id}: \"{user_question}\"")

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        wakili_instance = context.application.wakili_ai_instance

        final_response = await asyncio.to_thread(
            wakili_instance.get_wakili_response, user_question
        )

        # Create the inline keyboard for feedback
        keyboard = [
            [
                InlineKeyboardButton("👍 Helpful", callback_data="feedback_helpful"),
                InlineKeyboardButton("👎 Not Helpful", callback_data="feedback_not_helpful"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(final_response, parse_mode='Markdown', reply_markup=reply_markup)
        logger.info(f"Successfully sent final response to chat_id {chat_id}.")

    except Exception as e:
        logger.error(f"Error handling question from {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("I'm sorry, an internal error occurred. Please try again.")


# --- Initialization & Quart App ---

async def initialize_app():
    """Initializes and sets up the entire application."""
    global wakili_instance, application, TELEGRAM_TOKEN, WEBHOOK_URL
    # (Rest of initialization code is the same...)
    try:
        config = Config()
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        WEBHOOK_URL = os.getenv('WEBHOOK_URL')

        if not all([TELEGRAM_TOKEN, WEBHOOK_URL]):
            raise ValueError("Missing critical environment variables: TELEGRAM_BOT_TOKEN, WEBHOOK_URL")

        wakili_instance = WakiliAI(config)
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.wakili_ai_instance = wakili_instance

        # --- Add all new handlers ---
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_text_message))
        application.add_handler(MessageHandler(filters.ALL & ~filters.TEXT & ~filters.COMMAND, non_text_handler))
        application.add_handler(CallbackQueryHandler(feedback_handler))

        await application.initialize()
        await application.start()
        await application.bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}", allowed_updates=Update.ALL_TYPES)

        logger.info("Bot setup complete with all handlers. Webhook is set.")
        return True
    except Exception as e:
        logger.critical(f"FATAL: Application failed to initialize. Error: {e}", exc_info=True)
        return False


app = Quart(__name__)


@app.before_serving
async def startup():
    await initialize_app()


@app.route("/health")
async def health_check():
    return "OK", 200


@app.route(f"/{os.getenv('TELEGRAM_BOT_TOKEN', 'default')}", methods=["POST"])
async def webhook_handler():
    global application
    if not application: return "Internal Server Error", 500
    try:
        update = Update.de_json(await request.get_json(), application.bot)
        asyncio.create_task(application.process_update(update))
        return "ok"
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        return "Error", 500