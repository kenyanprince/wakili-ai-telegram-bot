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

HELP_MESSAGE = """*Wakili Wangu Bot Help* ⚖️\n\nI am an AI assistant for Kenyan law..."""
NON_TEXT_MESSAGE = "I'm sorry, I only understand text-based legal questions."


# --- THIS IS THE FIX: A FUNCTION TO SANITIZE MARKDOWN V2 ---
def sanitize_markdown_v2(text: str) -> str:
    """Escapes characters for Telegram's MarkdownV2 parser."""
    # Characters to escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
    # Note: We don't escape * and _ because we want them for bold/italics.
    # We will handle them separately.

    # First, handle characters that must be escaped.
    escape_chars = r'\[\]()~`>#+-=|{}.!'
    text = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    # Second, ensure that any * or _ that are NOT part of a matched pair
    # are also escaped. This is complex, so we'll do a simpler check for now:
    # If a line has an odd number of asterisks, escape them all on that line.
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        if line.count('*') % 2 != 0:
            line = line.replace('*', '\\*')
        if line.count('_') % 2 != 0:
            line = line.replace('_', '\\_')
        processed_lines.append(line)

    return '\n'.join(processed_lines)


# --- Core Bot Handlers ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(sanitize_markdown_v2(HELP_MESSAGE), parse_mode=ParseMode.MARKDOWN_V2)


async def non_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(NON_TEXT_MESSAGE)


async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    logger.info(f"Feedback from {query.from_user.id}: {query.data}")

    # Sanitize the original message text before editing
    sanitized_text = sanitize_markdown_v2(query.message.text)
    feedback_text = sanitize_markdown_v2("\n\n---\n*🙏 Thank you for your feedback!*")

    await query.edit_message_text(text=f"{sanitized_text}{feedback_text}", parse_mode=ParseMode.MARKDOWN_V2)


async def process_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Processing question from chat_id {chat_id}")
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        wakili_instance = context.application.wakili_ai_instance
        final_response_raw = await asyncio.to_thread(wakili_instance.get_response, user_question)

        # Sanitize the AI's response before sending
        final_response_sanitized = sanitize_markdown_v2(final_response_raw)

        keyboard = [[
            InlineKeyboardButton("👍 Helpful", callback_data="feedback_helpful"),
            InlineKeyboardButton("👎 Not Helpful", callback_data="feedback_not_helpful"),
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Use the more robust MarkdownV2 parser
        await update.message.reply_text(final_response_sanitized, parse_mode=ParseMode.MARKDOWN_V2,
                                        reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error handling question from {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("I'm sorry, an internal error occurred.")


# --- App Initialization & Server ---
app = Quart(__name__)
application = None


@app.before_serving
async def startup():
    global application
    # ... (rest of startup is the same)
    try:
        config = Config()
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        WEBHOOK_URL = os.getenv('WEBHOOK_URL')

        if not all([TELEGRAM_TOKEN, WEBHOOK_URL]):
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or WEBHOOK_URL environment variables.")

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