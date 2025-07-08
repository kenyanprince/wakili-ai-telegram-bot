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


# --- Enhanced Logging Setup ---
def setup_logging():
    """Configure comprehensive logging for the application."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    # Setup root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            # Uncomment below to also log to file
            # logging.FileHandler('wakili_bot.log')
        ]
    )

    # Set specific log levels for different components
    logging.getLogger('telegram').setLevel(logging.DEBUG)
    logging.getLogger('httpx').setLevel(logging.WARNING)  # Reduce HTTP request noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    return logging.getLogger(__name__)


logger = setup_logging()

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


# --- THIS IS THE FIX: A FUNCTION TO SANITIZE MARKDOWN V2 ---
def sanitize_markdown_v2(text: str) -> str:
    """Escapes characters for Telegram's MarkdownV2 parser while preserving bold/italics."""
    logger.debug("Sanitizing markdown text")

    # First, escape all special characters except for * and _
    escape_chars = r'\[\]()~`>#+-=|{}.!'
    sanitized_text = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    # Now, handle asterisks and underscores carefully
    # This is a simple approach: if a line has an odd number of them, escape them all.
    # A more complex regex could handle nested cases, but this covers most LLM errors.
    final_lines = []
    for line in sanitized_text.split('\n'):
        if line.count('*') % 2 != 0:
            line = line.replace('*', '\\*')
        if line.count('_') % 2 != 0:
            line = line.replace('_', '\\_')
        final_lines.append(line)

    sanitized_result = '\n'.join(final_lines)
    logger.debug(
        f"Markdown sanitization complete. Original length: {len(text)}, Sanitized length: {len(sanitized_result)}")
    return sanitized_result


# --- Core Bot Handlers ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_info = f"User: {update.effective_user.id} (@{update.effective_user.username})"
    chat_info = f"Chat: {update.effective_chat.id} ({update.effective_chat.type})"
    logger.info(f"Help command requested - {user_info}, {chat_info}")

    try:
        await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"Help message sent successfully to {update.effective_user.id}")
    except Exception as e:
        logger.error(f"Failed to send help message to {update.effective_user.id}: {e}")


async def non_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_info = f"User: {update.effective_user.id} (@{update.effective_user.username})"
    message_type = update.message.content_type if update.message else "unknown"
    logger.info(f"Non-text message received - {user_info}, Type: {message_type}")

    try:
        await update.message.reply_text(NON_TEXT_MESSAGE)
        logger.info(f"Non-text handler response sent to {update.effective_user.id}")
    except Exception as e:
        logger.error(f"Failed to send non-text response to {update.effective_user.id}: {e}")


async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_info = f"User: {query.from_user.id} (@{query.from_user.username})"
    feedback_type = query.data

    logger.info(f"Feedback received - {user_info}, Type: {feedback_type}")

    try:
        await query.answer()
        logger.debug(f"Callback query answered for {query.from_user.id}")

        # When editing, we must also use a compatible format.
        # We will simply edit the text and remove the buttons.
        await query.edit_message_text(
            text=f"{query.message.text}\n\n--- \n*🙏 Thank you for your feedback!*",
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info(f"Feedback acknowledgment sent to {query.from_user.id}")

    except Exception as e:
        logger.error(f"Error handling feedback from {query.from_user.id}: {e}", exc_info=True)


async def process_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    chat_id = update.message.chat_id
    user_info = f"User: {update.effective_user.id} (@{update.effective_user.username})"

    logger.info(f"Processing question from {user_info} in chat {chat_id}")
    logger.debug(f"Question text: {user_question[:100]}{'...' if len(user_question) > 100 else ''}")

    try:
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        logger.debug(f"Typing indicator sent to chat {chat_id}")

        # Get AI response
        wakili_instance = context.application.wakili_ai_instance
        logger.debug("Calling WakiliAI instance...")

        start_time = asyncio.get_event_loop().time()
        final_response_raw = await asyncio.to_thread(wakili_instance.get_response, user_question)
        end_time = asyncio.get_event_loop().time()

        processing_time = end_time - start_time
        logger.info(f"AI response generated in {processing_time:.2f} seconds for {user_info}")
        logger.debug(f"Raw AI response length: {len(final_response_raw)} characters")

        # Sanitize the AI's response before sending
        final_response_sanitized = sanitize_markdown_v2(final_response_raw)
        logger.debug(f"Response sanitized, final length: {len(final_response_sanitized)} characters")

        # Create feedback keyboard
        keyboard = [[
            InlineKeyboardButton("👍 Helpful", callback_data="feedback_helpful"),
            InlineKeyboardButton("👎 Not Helpful", callback_data="feedback_not_helpful"),
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Send response
        logger.debug(f"Sending response to {user_info}...")
        await update.message.reply_text(
            final_response_sanitized,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup
        )
        logger.info(f"Response sent successfully to {user_info}")

    except Exception as e:
        logger.error(f"Error handling question from {user_info} in chat {chat_id}: {e}", exc_info=True)

        try:
            await update.message.reply_text("I'm sorry, an internal error occurred while processing your request.")
            logger.info(f"Error message sent to {user_info}")
        except Exception as send_error:
            logger.error(f"Failed to send error message to {user_info}: {send_error}")


# --- App Initialization & Server ---
app = Quart(__name__)
application = None


@app.before_serving
async def startup():
    global application
    logger.info("Starting Wakili Wangu Telegram Bot initialization...")

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = Config()

        # Get environment variables
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        WEBHOOK_URL = os.getenv('WEBHOOK_URL')

        logger.info(f"Telegram token: {'✓ Set' if TELEGRAM_TOKEN else '✗ Missing'}")
        logger.info(f"Webhook URL: {'✓ Set' if WEBHOOK_URL else '✗ Missing'}")

        if not all([TELEGRAM_TOKEN, WEBHOOK_URL]):
            missing_vars = []
            if not TELEGRAM_TOKEN:
                missing_vars.append('TELEGRAM_BOT_TOKEN')
            if not WEBHOOK_URL:
                missing_vars.append('WEBHOOK_URL')
            raise ValueError(f"Missing critical environment variables: {', '.join(missing_vars)}")

        # Initialize WakiliAI
        logger.info("Initializing WakiliAI instance...")
        wakili_instance = WakiliAI(config)
        logger.info("WakiliAI instance created successfully")

        # Create Telegram application
        logger.info("Creating Telegram application...")
        ptb_app = Application.builder().token(TELEGRAM_TOKEN).build()
        ptb_app.wakili_ai_instance = wakili_instance
        logger.info("Telegram application created")

        # Add handlers
        logger.info("Adding message handlers...")
        ptb_app.add_handler(CommandHandler("help", help_command))
        ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_text_message))
        ptb_app.add_handler(MessageHandler(filters.ALL & ~filters.TEXT & ~filters.COMMAND, non_text_handler))
        ptb_app.add_handler(CallbackQueryHandler(feedback_handler))
        logger.info("All handlers added successfully")

        # Initialize and start application
        logger.info("Initializing Telegram application...")
        await ptb_app.initialize()
        logger.info("Starting Telegram application...")
        await ptb_app.start()

        # Set webhook
        webhook_url = f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}"
        logger.info(f"Setting webhook to: {webhook_url}")
        await ptb_app.bot.set_webhook(url=webhook_url, allowed_updates=Update.ALL_TYPES)
        logger.info("Webhook set successfully")

        application = ptb_app
        logger.info("🚀 Wakili Wangu Telegram Bot is now live and ready to serve!")

    except Exception as e:
        logger.critical(f"FATAL: Application failed to initialize. Error: {e}", exc_info=True)
        raise


@app.route("/health")
async def health_check():
    logger.debug("Health check endpoint accessed")
    return "OK", 200


@app.route(f"/{os.getenv('TELEGRAM_BOT_TOKEN', 'default')}", methods=["POST"])
async def webhook_handler():
    if application:
        try:
            logger.debug("Webhook request received")
            update_data = await request.get_json()
            logger.debug(f"Update data keys: {list(update_data.keys()) if update_data else 'None'}")

            update = Update.de_json(update_data, application.bot)
            logger.debug(f"Update object created: {update.update_id if update else 'None'}")

            asyncio.create_task(application.process_update(update))
            logger.debug("Update processing task created")

            return "ok"
        except Exception as e:
            logger.error(f"Error processing webhook: {e}", exc_info=True)
            return "Error", 500
    else:
        logger.error("Webhook called but application not initialized")
        return "Internal Server Error", 500


# --- Additional logging for debugging ---
@app.before_request
async def log_request_info():
    logger.debug(f"Request: {request.method} {request.path} from {request.remote_addr}")


@app.after_request
async def log_response_info(response):
    logger.debug(f"Response: {response.status_code} for {request.method} {request.path}")
    return response


if __name__ == "__main__":
    logger.info("Starting Wakili Wangu Bot server...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))