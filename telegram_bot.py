# telegram_bot.py

import logging
import os
import asyncio
import re
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
from telegram.constants import ParseMode
from wakili_engine import WakiliAI, Config

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HELP_MESSAGE = """
*Wakili Wangu Bot Help* ⚖️

I am an AI assistant for Kenyan law, designed to search statutes and case law to answer your questions.

*How to Use Me:*
Just ask a legal question in plain English, for example:
- "What are my rights if I am arrested?"
- "How do I register a company?"

I will provide a detailed, yet easy-to-understand explanation.

---
*_Disclaimer:_* _I am an AI, not a lawyer. My answers are for informational purposes only. Always consult a qualified advocate for your specific situation._
"""
NON_TEXT_MESSAGE = "I'm sorry, I can only understand legal questions written in text. Please type your question for me."


# --- Markdown Sanitizer ---
def sanitize_markdown_v2(text: str) -> str:
    """A simple function to escape characters for Telegram's MarkdownV2 parser."""
    escape_chars = r'\_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


# --- Core Bot Handlers ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.MARKDOWN)


async def non_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(NON_TEXT_MESSAGE)


async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    logger.info(f"Feedback from {query.from_user.id}: {query.data}")
    # We use query.message.text_markdown_v2 to get the text with entities
    await query.edit_message_text(
        text=f"{query.message.text_markdown_v2}\n\n---\n*🙏 Thank you for your feedback\!*",
        parse_mode=ParseMode.MARKDOWN_V2
    )


async def process_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Processing question from chat_id {chat_id}")
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        wakili_instance = context.application.wakili_ai_instance
        final_response_raw = await asyncio.to_thread(wakili_instance.get_response, user_question)

        # We don't need to sanitize if we trust the AI prompt to use safe Markdown.
        # Let's switch back to the more forgiving default parser.
        final_response = final_response_raw

        keyboard = [[
            InlineKeyboardButton("👍 Helpful", callback_data="feedback_helpful"),
            InlineKeyboardButton("👎 Not Helpful", callback_data="feedback_not_helpful"),
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(final_response, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error handling question from {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("I'm sorry, an internal error occurred.")


# --- App Initialization & Server ---
# THIS IS THE CRITICAL PART: 'app' must be in the global scope for Hypercorn
app = Quart(__name__)
application = None  # Define application in the global scope as well


@app.before_serving
async def startup():
    """This function is run by Quart ONCE before the server starts."""
    global application
    try:
        config = Config()
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        WEBHOOK_URL = os.getenv('WEBHOOK_URL')

        if not all([TELEGRAM_TOKEN, WEBHOOK_URL]):
            raise ValueError("Missing critical environment variables.")

        wakili_instance = WakiliAI(config)

        ptb_app = Application.builder().token(TELEGRAM_TOKEN).build()
        ptb_app.wakili_ai_instance = wakili_instance

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