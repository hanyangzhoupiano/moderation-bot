import os
import threading

from flask import Flask, jsonify
from waitress import serve

app = Flask(__name__)

bot = None


@app.route("/")
def home():
    return "Discord bot is running!", 200


@app.route("/status")
def status():
    if bot is None:
        return jsonify({
            "status": "starting",
            "message": "Bot has not been initialized yet."
        }), 503

    if not bot.is_ready():
        return jsonify({
            "status": "starting",
            "message": "Discord bot is connecting."
        }), 503

    return jsonify({
        "status": "online",
        "bot": str(bot.user),
        "bot_id": bot.user.id,
        "latency": round(bot.latency * 1000, 2)
    }), 200


def run_web_server():
    port = int(os.environ.get("PORT", 10000))

    serve(
        app,
        host="0.0.0.0",
        port=port
    )


def start_web_server(discord_bot):
    global bot
    bot = discord_bot

    thread = threading.Thread(
        target=run_web_server,
        daemon=True
    )

    thread.start()
