
import os
import re
import json
import asyncio
import discord

from datetime import datetime, timedelta, timezone
from discord.ext import commands
from groq import AsyncGroq
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing.")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing.")

MODEL = "openai/gpt-oss-120b"

DEFAULT_REASON = "No reason provided."

MAX_TIMEOUT_DAYS = 28
MAX_TOOL_CALLS = 6
MAX_TOOL_ROUNDS = 6

# Maximum number of conversation messages stored per moderator.
# This does not include the system prompt.
MAX_HISTORY_MESSAGES = 30

groq = AsyncGroq(api_key=GROQ_API_KEY)


# ============================================================
# DISCORD SETUP
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    help_command=None
)


# ============================================================
# PER-USER CONVERSATION HISTORY
# ============================================================

# Each moderator gets an independent conversation history.
#
# Example:
# {
#     123456789: [message1, message2, ...],
#     987654321: [message1, message2, ...]
# }

conversation_histories = {}

# Prevent simultaneous requests from modifying the same history.
user_locks = {}


def get_user_lock(user_id):
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()

    return user_locks[user_id]


def get_user_history(user_id):
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []

    return conversation_histories[user_id]


def trim_history(history):
    """
    Keep the most recent conversation messages.

    Tool calls and their results must stay together, so we trim
    complete exchanges rather than arbitrary individual messages.
    """

    if len(history) <= MAX_HISTORY_MESSAGES:
        return history

    history = history[-MAX_HISTORY_MESSAGES:]

    # Avoid starting history with an orphaned tool result.
    while history and history[0].get("role") == "tool":
        history.pop(0)

    # If the first assistant message contains tool calls, make sure
    # we don't retain it without the corresponding tool responses.
    while history and history[0].get("role") == "assistant":
        if history[0].get("tool_calls"):
            history.pop(0)
        else:
            break

    return history


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a friendly, intelligent Discord moderation assistant.

You are talking directly to a server moderator through Discord.

Your purpose is to help moderators with:
- Moderation actions
- Moderation advice
- Discussing individual members and situations
- Understanding server rules
- Handling difficult moderation situations
- General Discord questions
- Casual conversation and entertainment

You are also capable of executing moderation actions using the
tools provided to you.

IMPORTANT: YOU ARE A CONVERSATIONAL AI FIRST.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CONVERSATION VS MODERATION ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Not every message is an instruction to moderate someone.

Moderators can freely talk to you about users, incidents,
potential punishments, server drama, moderation decisions,
or anything else without you automatically taking action.

You MUST distinguish between:

A. EXPLICIT MODERATION INSTRUCTIONS
B. DISCUSSION ABOUT MODERATION
C. QUESTIONS ABOUT MODERATION
D. NORMAL CONVERSATION

Only category A allows you to call moderation tools.

Examples of EXPLICIT MODERATION INSTRUCTIONS:

- "Ban John."
- "Please ban @John for harassment."
- "Kick this user: 123456789."
- "Timeout John for 2 hours."
- "Softban @John for spamming."
- "Unban 123456789."
- "Remove John's timeout."

These are direct instructions to perform an action.

Examples of DISCUSSION ABOUT MODERATION:

- "I think John deserves a ban."
- "John has been causing problems lately."
- "I'm considering banning John."
- "What do you think about banning John?"
- "Should I kick this person?"
- "John might need a timeout."
- "I'm not sure whether to ban or kick John."
- "Would a softban be appropriate here?"

These are discussions, NOT instructions.

Do not call moderation tools for these messages.

Instead, discuss the situation and provide helpful advice.

Examples of QUESTIONS:

- "What does a softban do?"
- "How does Discord timeout work?"
- "Can you explain the difference between a kick and a ban?"
- "Can you ban someone?"
- "What happens when someone gets banned?"

Answer normally. Never execute tools just because a
moderation action is mentioned in a question.

Examples of NORMAL CONVERSATION:

- "Hey!"
- "How are you?"
- "Tell me a joke."
- "What's your opinion on this server?"
- "I'm bored."
- "Let's talk about moderation."

Respond naturally and conversationally.

NEVER assume that mentioning a user means the moderator wants
that user punished.

NEVER assume that discussing a potential punishment means
the moderator has approved that punishment.

NEVER execute a moderation action based on speculation,
hypothetical scenarios, suggestions, or your own judgment.

When uncertain whether the moderator is explicitly instructing
you to perform an action, ASK FOR CLARIFICATION.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. CONVERSATION MEMORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have access to previous messages from this moderator.

Use conversation history to understand:
- Previous discussions
- User references
- Context from earlier messages
- Follow-up questions
- Previously discussed moderation situations

For example:

Moderator: "John keeps spamming."
You: "Have you warned him?"

Moderator: "Yes, twice."
You: "Then a timeout might be appropriate."

Moderator: "Give him one for 2 hours."

You should understand that "him" refers to John.

However, conversation history is NOT authorization to
perform moderation actions.

A previous discussion about banning someone does not mean
you should automatically ban them later.

The moderator must still explicitly instruct you to perform
the action.

If a target is ambiguous, ask for clarification.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. AVAILABLE MODERATION TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BAN:
Permanently bans a user from the server.

UNBAN:
Removes an existing ban from a user.

SOFTBAN:
Bans a user and immediately unbans them.
Deletes up to seven days of recent messages.

KICK:
Removes a member from the server without banning them.

TIMEOUT:
Prevents a member from communicating and participating
in certain server activities for a specified duration.

Duration is optional. If omitted, defaults to 28 days.

Maximum duration is 28 days.

Supported formats:
- 30s
- 10m
- 2h
- 3d
- 1w
- 1d12h
- 2h30m

UNTIMEOUT:
Manually removes an existing timeout from a member.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. CHOOSING THE CORRECT ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Examples:

"Ban John" -> BAN
"Permanently ban John" -> BAN
"Kick John" -> KICK
"Remove John from the server" -> Ask whether they mean kick or ban.
"Softban John" -> SOFTBAN
"Unban John" -> UNBAN
"Timeout John for 2 hours" -> TIMEOUT
"Mute John for 30 minutes" -> TIMEOUT
"Timeout John" -> TIMEOUT, defaulting to 28 days.
"Untimeout John" -> UNTIMEOUT
"Unmute John" -> UNTIMEOUT

Never confuse these actions.

A kick is not a ban.
A softban is not a permanent ban.
A timeout is not a kick.
An unban does not automatically make someone rejoin.
An untimeout only removes the timeout.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. USER IDENTIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Users can be identified by:
- Discord mention
- Discord user ID
- Username
- Display name
- Previous conversation context

Preserve the exact user identifier supplied by the moderator
when calling a tool.

If multiple users match, request a mention or ID.

Never guess the target.

Never accidentally moderate the moderator.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. REASONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every moderation action must have a reason.

If a reason is provided, preserve its meaning.

If no reason is provided, use exactly:

"No reason provided."

Never invent offenses, allegations, evidence, or reasons.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. TOOL EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Only call moderation tools when the moderator clearly and
explicitly instructs you to perform an action.

Before calling a tool, identify:
- Requested action
- Target user
- Duration, if applicable
- Reason, if provided

Do not ask unnecessary questions when all required information
is already available.

Multiple actions are allowed if explicitly requested.

Never perform additional actions that were not requested.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. TOOL RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tool results are authoritative.

If success is true:
- Confirm the action succeeded.
- Identify the affected user.
- State the action.
- Include the duration when applicable.

If success is false:
- Explain the failure.
- Never claim the action succeeded.

Never fabricate tool results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. GENERAL BEHAVIOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are allowed to be friendly, funny, casual, and conversational.

You can:
- Tell jokes.
- Discuss moderation decisions.
- Give opinions and advice.
- Help evaluate punishments.
- Explain Discord features.
- Discuss server rules.
- Talk about members without taking action.
- Have normal conversations.

When discussing a member, remain objective and do not assume
accusations are true.

Do not encourage unfair or abusive moderation.

The human moderator makes the decisions.

You are an assistant, not an autonomous moderator.

Be concise unless asked for more detail.
"""


# ============================================================
# GROQ TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ban",
            "description": (
                "Permanently ban a user from the Discord server. "
                "Use only when explicitly instructed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "User mention, ID, or username."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Ban reason."
                    }
                },
                "required": ["user"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unban",
            "description": "Unban a previously banned user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "User ID, mention, or banned username."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Unban reason."
                    }
                },
                "required": ["user"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "softban",
            "description": (
                "Ban and immediately unban a user, deleting up to "
                "seven days of recent messages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "User mention, ID, or username."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Softban reason."
                    }
                },
                "required": ["user"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kick",
            "description": "Kick a member without banning them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "User mention, ID, or username."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Kick reason."
                    }
                },
                "required": ["user"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "timeout",
            "description": (
                "Timeout a member. Duration is optional and "
                "defaults to 28 days. Maximum is 28 days."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "User mention, ID, or username."
                    },
                    "duration": {
                        "type": "string",
                        "description": "Optional duration."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Timeout reason."
                    }
                },
                "required": ["user"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "untimeout",
            "description": "Remove an existing timeout from a member.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "User mention, ID, or username."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for removing the timeout."
                    }
                },
                "required": ["user"]
            }
        }
    }
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def make_result(success, message, **extra):
    return {
        "success": success,
        "message": message,
        **extra
    }


def get_reason(reason):
    if not reason or not reason.strip():
        return DEFAULT_REASON

    return reason.strip()


def parse_duration(value):
    if not value:
        raise ValueError("Duration cannot be empty.")

    value = value.lower().strip()

    pattern = r"(\d+)\s*(w|d|h|m|s)"
    matches = list(re.finditer(pattern, value))

    if not matches:
        raise ValueError(
            "Invalid duration. Examples: 10m, 2h, 3d, 1w."
        )

    remainder = re.sub(pattern, "", value)

    if remainder.strip():
        raise ValueError(f"Invalid duration: {value}")

    units = {
        "w": 604800,
        "d": 86400,
        "h": 3600,
        "m": 60,
        "s": 1
    }

    seconds = sum(
        int(match.group(1)) * units[match.group(2)]
        for match in matches
    )

    if seconds <= 0:
        raise ValueError("Duration must be greater than zero.")

    return timedelta(seconds=seconds)


def parse_user_id(value):
    value = value.strip()

    match = re.fullmatch(r"<@!?(\d+)>", value)

    if match:
        return int(match.group(1))

    if value.isdigit():
        return int(value)

    return None


# ============================================================
# USER RESOLUTION
# ============================================================

async def resolve_member(guild, value):

    user_id = parse_user_id(value)

    if user_id:
        member = guild.get_member(user_id)

        if member:
            return member

        try:
            return await guild.fetch_member(user_id)
        except discord.NotFound:
            return None

    value = value.strip().lower()
    matches = []

    for member in guild.members:
        names = {
            member.name.lower(),
            member.display_name.lower(),
            str(member).lower()
        }

        if member.global_name:
            names.add(member.global_name.lower())

        if value in names:
            matches.append(member)

    if len(matches) > 1:
        raise ValueError(
            f"Multiple members match '{value}'. "
            "Please provide their mention or ID."
        )

    return matches[0] if matches else None


async def resolve_user(guild, value):

    user_id = parse_user_id(value)

    if user_id:
        member = guild.get_member(user_id)

        if member:
            return member

        try:
            return await bot.fetch_user(user_id)
        except discord.NotFound:
            return None

    return await resolve_member(guild, value)


async def resolve_banned_user(guild, value):

    user_id = parse_user_id(value)

    if user_id:
        return discord.Object(id=user_id)

    matches = []

    async for entry in guild.bans(limit=None):
        user = entry.user

        names = {
            user.name.lower(),
            str(user).lower()
        }

        if user.global_name:
            names.add(user.global_name.lower())

        if value.lower() in names:
            matches.append(user)

    if len(matches) > 1:
        raise ValueError(
            f"Multiple banned users match '{value}'. "
            "Please provide their ID."
        )

    return matches[0] if matches else None


# ============================================================
# MODERATION: BAN
# ============================================================

async def ban_user(guild, user, reason=None):
    reason = get_reason(reason)

    try:
        target = await resolve_user(guild, user)

        if target is None:
            return make_result(False, f"Could not find user '{user}'.")

        if target.id == guild.owner_id:
            return make_result(False, "Cannot ban the server owner.")

        if target.id == bot.user.id:
            return make_result(False, "Cannot ban the bot itself.")

        await guild.ban(target, reason=reason)

        return make_result(
            True,
            f"Successfully permanently banned {target}.",
            user_id=target.id,
            reason=reason
        )

    except discord.Forbidden:
        return make_result(False, "Discord denied the ban. Check permissions and role hierarchy.")
    except discord.HTTPException as e:
        return make_result(False, f"Discord API error: {e}")
    except Exception as e:
        return make_result(False, f"Unexpected error: {e}")


# ============================================================
# MODERATION: UNBAN
# ============================================================

async def unban_user(guild, user, reason=None):
    reason = get_reason(reason)

    try:
        target = await resolve_banned_user(guild, user)

        if target is None:
            return make_result(False, f"Could not find banned user '{user}'.")

        await guild.unban(target, reason=reason)

        return make_result(
            True,
            f"Successfully unbanned user ID {target.id}.",
            user_id=target.id,
            reason=reason
        )

    except discord.NotFound:
        return make_result(False, "This user is not banned.")
    except discord.Forbidden:
        return make_result(False, "Discord denied the unban. Check bot permissions.")
    except discord.HTTPException as e:
        return make_result(False, f"Discord API error: {e}")
    except Exception as e:
        return make_result(False, f"Unexpected error: {e}")


# ============================================================
# MODERATION: SOFTBAN
# ============================================================

async def softban_user(guild, user, reason=None):
    reason = get_reason(reason)

    try:
        target = await resolve_user(guild, user)

        if target is None:
            return make_result(False, f"Could not find user '{user}'.")

        if target.id == guild.owner_id:
            return make_result(False, "Cannot softban the server owner.")

        if target.id == bot.user.id:
            return make_result(False, "Cannot softban the bot itself.")

        await guild.ban(
            target,
            reason=reason,
            delete_message_seconds=604800
        )

        try:
            await guild.unban(
                target,
                reason=f"Softban completed: {reason}"
            )
        except discord.HTTPException as e:
            return make_result(
                False,
                "Ban succeeded but unban failed. The user may still be banned. "
                f"Error: {e}",
                user_id=target.id,
                partially_completed=True
            )

        return make_result(
            True,
            f"Successfully softbanned {target}.",
            user_id=target.id,
            reason=reason
        )

    except discord.Forbidden:
        return make_result(False, "Discord denied the softban. Check permissions and role hierarchy.")
    except discord.HTTPException as e:
        return make_result(False, f"Discord API error: {e}")
    except Exception as e:
        return make_result(False, f"Unexpected error: {e}")


# ============================================================
# MODERATION: KICK
# ============================================================

async def kick_user(guild, user, reason=None):
    reason = get_reason(reason)

    try:
        target = await resolve_member(guild, user)

        if target is None:
            return make_result(False, f"Could not find server member '{user}'.")

        if target.id == guild.owner_id:
            return make_result(False, "Cannot kick the server owner.")

        if target.id == bot.user.id:
            return make_result(False, "Cannot kick the bot itself.")

        await target.kick(reason=reason)

        return make_result(
            True,
            f"Successfully kicked {target}.",
            user_id=target.id,
            reason=reason
        )

    except discord.Forbidden:
        return make_result(False, "Discord denied the kick. Check permissions and role hierarchy.")
    except discord.HTTPException as e:
        return make_result(False, f"Discord API error: {e}")
    except Exception as e:
        return make_result(False, f"Unexpected error: {e}")


# ============================================================
# MODERATION: TIMEOUT
# ============================================================

async def timeout_user(guild, user, duration=None, reason=None):
    reason = get_reason(reason)

    try:
        target = await resolve_member(guild, user)

        if target is None:
            return make_result(False, f"Could not find server member '{user}'.")

        if target.id == guild.owner_id:
            return make_result(False, "Cannot timeout the server owner.")

        if target.id == bot.user.id:
            return make_result(False, "Cannot timeout the bot itself.")

        if duration:
            try:
                parsed_duration = parse_duration(duration)
            except ValueError as e:
                return make_result(False, str(e))
        else:
            parsed_duration = timedelta(days=MAX_TIMEOUT_DAYS)

        if parsed_duration > timedelta(days=MAX_TIMEOUT_DAYS):
            return make_result(False, "Discord timeouts cannot exceed 28 days.")

        until = datetime.now(timezone.utc) + parsed_duration

        await target.timeout(until, reason=reason)

        return make_result(
            True,
            f"Successfully timed out {target}.",
            user_id=target.id,
            duration=str(parsed_duration),
            expires_at=until.isoformat(),
            reason=reason
        )

    except discord.Forbidden:
        return make_result(False, "Discord denied the timeout. Check permissions and role hierarchy.")
    except discord.HTTPException as e:
        return make_result(False, f"Discord API error: {e}")
    except Exception as e:
        return make_result(False, f"Unexpected error: {e}")


# ============================================================
# MODERATION: UNTIMEOUT
# ============================================================

async def untimeout_user(guild, user, reason=None):
    reason = get_reason(reason)

    try:
        target = await resolve_member(guild, user)

        if target is None:
            return make_result(False, f"Could not find server member '{user}'.")

        await target.timeout(None, reason=reason)

        return make_result(
            True,
            f"Successfully removed the timeout from {target}.",
            user_id=target.id,
            reason=reason
        )

    except discord.Forbidden:
        return make_result(False, "Discord denied the untimeout. Check permissions and role hierarchy.")
    except discord.HTTPException as e:
        return make_result(False, f"Discord API error: {e}")
    except Exception as e:
        return make_result(False, f"Unexpected error: {e}")


# ============================================================
# TOOL EXECUTION
# ============================================================

async def execute_tool(name, arguments, guild):

    handlers = {
        "ban": ban_user,
        "unban": unban_user,
        "softban": softban_user,
        "kick": kick_user,
        "timeout": timeout_user,
        "untimeout": untimeout_user
    }

    handler = handlers.get(name)

    if handler is None:
        return make_result(False, f"Unknown tool: {name}")

    try:
        return await handler(guild=guild, **arguments)

    except TypeError as e:
        return make_result(False, f"Invalid arguments: {e}")

    except Exception as e:
        return make_result(False, f"Tool execution failed: {e}")


# ============================================================
# AI CONVERSATION WITH PER-USER HISTORY
# ============================================================

async def ask_ai(ctx, user_message):

    user_id = ctx.author.id

    # Lock this moderator's history so simultaneous requests
    # from the same moderator cannot corrupt the conversation.
    async with get_user_lock(user_id):

        history = get_user_history(user_id)

        # Add the new user message.
        history.append({
            "role": "user",
            "content": (
                f"Discord server: {ctx.guild.name}\n"
                f"Server ID: {ctx.guild.id}\n"
                f"Moderator: {ctx.author} (ID: {user_id})\n\n"
                f"Message:\n{user_message}"
            )
        })

        # Keep history within limits.
        conversation_histories[user_id] = trim_history(history)
        history = conversation_histories[user_id]

        # Build the request with the system prompt.
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ] + list(history)

        tool_calls_executed = 0

        for _ in range(MAX_TOOL_ROUNDS):

            response = await groq.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=2000
            )

            assistant_message = response.choices[0].message

            assistant_data = assistant_message.model_dump(
                exclude_none=True
            )

            messages.append(assistant_data)

            # The AI has finished responding.
            if not assistant_message.tool_calls:

                final_response = (
                    assistant_message.content or "I'm not sure what to say!"
                )

                # Save the complete conversation, including tool calls.
                history.extend(messages[len(history) + 1:])

                conversation_histories[user_id] = trim_history(history)

                return final_response

            # Execute requested tools.
            for tool_call in assistant_message.tool_calls:

                if tool_calls_executed >= MAX_TOOL_CALLS:

                    tool_result = make_result(
                        False,
                        "Maximum tool execution limit reached."
                    )

                else:
                    tool_calls_executed += 1

                    try:
                        tool_name = tool_call.function.name

                        arguments = json.loads(
                            tool_call.function.arguments
                        )

                        tool_result = await execute_tool(
                            tool_name,
                            arguments,
                            ctx.guild
                        )

                    except json.JSONDecodeError:
                        tool_result = make_result(
                            False,
                            "Invalid JSON arguments."
                        )

                    except Exception as e:
                        tool_result = make_result(
                            False,
                            f"Tool execution error: {e}"
                        )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })

        # If the model reaches its limits, save what happened.
        history.extend(messages[len(history) + 1:])

        conversation_histories[user_id] = trim_history(history)

        return (
            "I reached the maximum number of tool/reasoning rounds. "
            "Please try a simpler request."
        )


# ============================================================
# !CHAT COMMAND
# ============================================================

@bot.command(name="chat")
@commands.guild_only()
async def chat(ctx, *, message: str = None):

    if not ctx.author.guild_permissions.ban_members:
        await ctx.reply(
            "You need the **Ban Members** permission to use this command.",
            mention_author=False
        )
        return

    if not message or not message.strip():
        await ctx.reply(
            "Please provide a message!\n"
            "Example: `!chat Hello, how are you?`",
            mention_author=False
        )
        return

    async with ctx.typing():

        try:
            response = await ask_ai(
                ctx,
                message.strip()
            )

        except Exception as e:
            print(f"[GROQ ERROR] {repr(e)}")

            await ctx.reply(
                "I encountered an error while contacting the AI. "
                "Please try again later.",
                mention_author=False
            )
            return

    response = str(response)

    # Discord's message limit is 2000 characters.
    chunks = [
        response[i:i + 1900]
        for i in range(0, len(response), 1900)
    ]

    for chunk in chunks:
        await ctx.reply(
            chunk,
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none()
        )


# ============================================================
# !RESET COMMAND
# ============================================================

@bot.command(name="reset")
@commands.guild_only()
async def reset(ctx):

    if not ctx.author.guild_permissions.ban_members:
        await ctx.reply(
            "You need the **Ban Members** permission to use this command.",
            mention_author=False
        )
        return

    user_id = ctx.author.id

    async with get_user_lock(user_id):

        existed = user_id in conversation_histories

        conversation_histories.pop(user_id, None)

    if existed:
        await ctx.reply(
            "Your conversation history has been reset. "
            "We can start fresh!",
            mention_author=False
        )
    else:
        await ctx.reply(
            "You don't have any saved conversation history yet.",
            mention_author=False
        )


# ============================================================
# EVENTS
# ============================================================

@bot.event
async def on_ready():

    print("=" * 50)
    print(f"Logged in as: {bot.user}")
    print(f"Bot ID: {bot.user.id}")
    print(f"AI Model: {MODEL}")
    print("Commands: !chat, !reset")
    print("Moderation tools: 6")
    print("Temporary bans: Disabled")
    print("Per-moderator conversation history: Enabled")
    print("AI assistant is ready!")
    print("=" * 50)


# ============================================================
# START BOT
# ============================================================

bot.run(DISCORD_TOKEN)
