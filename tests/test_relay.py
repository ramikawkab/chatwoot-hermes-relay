import asyncio
import base64
import hashlib
import hmac
import io
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as relay

def picture(fmt: str) -> bytes:
    out = io.BytesIO()
    Image.new("RGBA" if fmt != "JPEG" else "RGB", (8, 8), (255, 0, 0)).save(out, fmt)
    return out.getvalue()


PHOTO = picture("JPEG")
PNG = picture("PNG")
VOICE = b"OggS" + b"opus-bytes" * 20
CHAT = "https://chat.example.com"
PHOTO_URL = CHAT + "/rails/active_storage/blobs/redirect/abc/photo.jpg"
VOICE_URL = CHAT + "/rails/active_storage/blobs/redirect/def/voice.ogg"


class Services:
    """Stands in for Chatwoot, Hermes, OpenAI and an image host, and records what the relay sent."""

    def __init__(self):
        self.hermes_reply = "Hello!"
        self.hermes_status = 200
        self.hermes_delay = 0.0
        self.transcript = "I want a book for Noah"
        self.transcribe_status = 200
        self.hermes_calls = []  # (headers, json body)
        self.transcribe_calls = []  # raw multipart bodies
        self.sent = []  # what reached the Chatwoot conversation, in order
        self.fetched = []  # (address connected to, Host header) for every download
        self.files = {
            PHOTO_URL: (PHOTO, "image/jpeg"),
            VOICE_URL: (VOICE, "audio/ogg"),
            "https://cdn.example.com/p.png": (PNG, "image/png"),
            "https://cdn.example.com/p.webp": (picture("WEBP"), "image/webp"),
            "https://cdn.example.com/p.svg": (b"<svg></svg>", "image/svg+xml"),
            "https://cdn.example.com/page.html": (b"<html></html>", "text/html"),
        }

    async def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.headers.get("host", request.url.host)
        url = f"{request.url.scheme}://{host}{request.url.raw_path.decode()}"
        if host.startswith("hermes"):
            self.hermes_calls.append((request.headers, json.loads(request.content)))
            await asyncio.sleep(self.hermes_delay)
            if self.hermes_status != 200:
                return httpx.Response(self.hermes_status, json={"error": "down"})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": self.hermes_reply}}]},
                headers={"X-Hermes-Session-Id": request.headers.get("X-Hermes-Session-Id", "")},
            )
        if host == "api.openai.com":
            self.transcribe_calls.append(request.content)
            if self.transcribe_status != 200:
                return httpx.Response(self.transcribe_status, json={"error": "nope"})
            return httpx.Response(200, json={"text": self.transcript})
        if request.method == "POST" and url.endswith("/toggle_status"):
            self.sent.append({"status": json.loads(request.content)["status"], "url": url})
            return httpx.Response(200, json={})
        if request.method == "POST" and url.endswith("/messages"):
            if request.headers["content-type"].startswith("multipart/form-data"):
                self.sent.append({"attachment": request.content, "url": url})
            else:
                body = json.loads(request.content)
                self.sent.append({"text": body["content"], "private": body["private"], "url": url})
            return httpx.Response(200, json={})
        self.fetched.append((request.url.host, host))
        if url in self.files:
            data, ctype = self.files[url]
            return httpx.Response(200, content=data, headers={"content-type": ctype})
        return httpx.Response(404)

    # What the customer saw (public messages and attachments, no private notes)
    def customer_texts(self):
        return [m["text"] for m in self.sent if "text" in m and not m["private"]]

    def private_notes(self):
        return [m["text"] for m in self.sent if "text" in m and m["private"]]

    def attachments(self):
        return [m["attachment"] for m in self.sent if "attachment" in m]

    def last_user_content(self):
        return self.hermes_calls[-1][1]["messages"][-1]["content"]


ADDRESSES = {
    "chat.example.com": ["31.97.52.160"],
    "cdn.example.com": ["93.184.216.34"],
    "internal.example.com": ["10.0.0.5"],
}


@pytest.fixture
def services(monkeypatch):
    s = Services()
    monkeypatch.setattr(relay, "http", httpx.AsyncClient(transport=httpx.MockTransport(s.handler)))
    monkeypatch.setattr(relay, "_resolve", lambda host: ADDRESSES.get(host, ["127.0.0.1"]))
    yield s
    for task in list(relay.running):
        task.cancel()
    relay.pending.clear()
    relay.timers.clear()
    relay.handed_off_at.clear()


def message(content="", attachments=None, conversation_id=7, account_id=1):
    return {
        "event": "message_created",
        "message_type": "incoming",
        "private": False,
        "content": content,
        "attachments": attachments or [],
        "account": {"id": account_id},
        "conversation": {"id": conversation_id, "status": "pending"},
    }


def photo():
    return {"file_type": "image", "data_url": PHOTO_URL, "extension": "jpg"}


def voice():
    return {"file_type": "audio", "data_url": VOICE_URL, "extension": None}


def parts(content, kind):
    return [p for p in content if p["type"] == kind]


def text_of(content):
    return content if isinstance(content, str) else parts(content, "text")[0]["text"]


# ----------------------------------------------------------------------------
# Photos and voice notes from the customer
# ----------------------------------------------------------------------------
async def test_photo_reaches_hermes_as_an_image_with_its_link(services):
    await relay.handle_batch([message(attachments=[photo()])])

    content = services.last_user_content()
    image = parts(content, "image_url")[0]["image_url"]["url"]
    assert image.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image.split(",", 1)[1]) == PHOTO
    assert PHOTO_URL in text_of(content)
    assert services.customer_texts() == ["Hello!"]


async def test_photo_with_a_caption_keeps_both(services):
    await relay.handle_batch([message("This is Noah", attachments=[photo()])])

    content = services.last_user_content()
    assert "This is Noah" in text_of(content)
    assert len(parts(content, "image_url")) == 1


async def test_photo_served_without_an_image_type_is_still_sent_as_an_image(services):
    services.files[PHOTO_URL] = (PHOTO, "application/octet-stream")
    await relay.handle_batch([message(attachments=[photo()])])

    image = parts(services.last_user_content(), "image_url")[0]["image_url"]["url"]
    assert image.startswith("data:image/jpeg;base64,")


async def test_photos_over_the_size_budget_go_as_links_only(services, monkeypatch):
    monkeypatch.setattr(relay, "MAX_INLINE_IMAGE_BYTES", len(PHOTO) + 1)
    await relay.handle_batch([message(attachments=[photo(), photo()])])

    content = services.last_user_content()
    assert len(parts(content, "image_url")) == 1
    assert "Photo 2 from the customer, too large to show here" in text_of(content)


async def test_voice_note_reaches_hermes_as_text(services):
    await relay.handle_batch([message(attachments=[voice()])])

    upload = services.transcribe_calls[0]
    assert b"gpt-4o-transcribe" in upload and VOICE in upload and b'filename="voice.ogg"' in upload
    assert services.last_user_content() == "(voice note) I want a book for Noah"


async def test_voice_note_that_cannot_be_transcribed_asks_to_type(services):
    services.transcribe_status = 500
    await relay.handle_batch([message(attachments=[voice()])])

    assert services.hermes_calls == []
    assert services.customer_texts() == [relay.VOICE_ERROR_MESSAGE]


async def test_a_silent_voice_note_asks_to_type(services):
    services.transcript = ""
    await relay.handle_batch([message(attachments=[voice()])])

    assert services.hermes_calls == []
    assert services.customer_texts() == [relay.VOICE_ERROR_MESSAGE]


async def test_asking_for_a_person_in_a_voice_note_hands_off(services):
    services.transcript = "Can I talk to a real person please"
    await relay.handle_batch([message(attachments=[voice()])])

    assert services.hermes_calls == []
    assert services.customer_texts() == [relay.HANDOFF_MESSAGE]
    assert any(m.get("status") == "open" for m in services.sent)


async def test_a_file_the_relay_cannot_read_gets_the_fixed_reply(services):
    await relay.handle_batch([message(attachments=[{"file_type": "video", "data_url": CHAT + "/v.mp4"}])])

    assert services.hermes_calls == []
    assert services.customer_texts() == [relay.ATTACHMENT_MESSAGE]


# ----------------------------------------------------------------------------
# Conversation memory lives in Hermes, one session per Chatwoot conversation
# ----------------------------------------------------------------------------
async def test_each_turn_sends_only_the_new_message_with_the_session_id(services):
    await relay.handle_batch([message("Hi")])
    await relay.handle_batch([message("Noah is 5")])

    headers, body = services.hermes_calls[-1]
    assert headers["X-Hermes-Session-Id"] == "chatwoot-1-7"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][-1]["content"] == "Noah is 5"


async def test_messages_sent_in_quick_succession_are_answered_once(services):
    relay.enqueue(message("Hi"))
    relay.enqueue(message(attachments=[photo()]))
    await asyncio.sleep(0.3)

    assert len(services.hermes_calls) == 1
    content = services.last_user_content()
    assert "Hi" in text_of(content)
    assert len(parts(content, "image_url")) == 1
    assert services.customer_texts() == ["Hello!"]


async def test_each_new_message_restarts_the_wait(services, monkeypatch):
    monkeypatch.setattr(relay, "BATCH_SECONDS", 0.2)
    relay.enqueue(message("Hi"))
    await asyncio.sleep(0.15)
    relay.enqueue(message("This is Noah"))
    await asyncio.sleep(0.1)  # 0.25 s after the first message, 0.1 s after the second

    assert services.hermes_calls == []
    await asyncio.sleep(0.3)
    assert len(services.hermes_calls) == 1
    assert services.last_user_content() == "Hi\nThis is Noah"


async def test_messages_sent_during_a_long_answer_are_answered_together(services):
    services.hermes_delay = 0.3
    relay.enqueue(message("a"))
    await asyncio.sleep(0.1)
    relay.enqueue(message("b"))
    await asyncio.sleep(0.1)
    relay.enqueue(message("c"))
    await asyncio.sleep(0.9)

    assert len(services.hermes_calls) == 2
    assert services.last_user_content() == "b\nc"


async def test_two_accounts_with_the_same_conversation_number_stay_apart(services):
    relay.enqueue(message("one", account_id=1))
    relay.enqueue(message("two", account_id=2))
    await asyncio.sleep(0.3)

    sessions = sorted(h["X-Hermes-Session-Id"] for h, _ in services.hermes_calls)
    assert sessions == ["chatwoot-1-7", "chatwoot-2-7"]
    assert sorted(m["url"].split("/accounts/")[1][0] for m in services.sent) == ["1", "2"]


async def test_queued_messages_are_answered_when_the_relay_shuts_down(services, monkeypatch):
    monkeypatch.setattr(relay, "BATCH_SECONDS", 30)
    relay.enqueue(message("Hi"))
    await relay.answer_queued_now(grace=2)

    assert len(services.hermes_calls) == 1
    assert services.customer_texts() == ["Hello!"]


# ----------------------------------------------------------------------------
# Handing over to a person
# ----------------------------------------------------------------------------
async def test_handoff_marker_sends_the_reply_and_opens_the_conversation(services):
    services.hermes_reply = "Let me get a colleague. [[HANDOFF]]"
    await relay.handle_batch([message("I want a refund")])

    assert services.customer_texts() == ["Let me get a colleague."]
    assert services.sent[-1]["status"] == "open"


async def test_messages_queued_before_a_handoff_are_not_answered(services):
    services.hermes_delay = 0.2
    services.hermes_reply = "A colleague will help. [[HANDOFF]]"
    relay.enqueue(message("I want a refund"))
    await asyncio.sleep(0.1)
    relay.enqueue(message("Hello??"))  # arrives while Hermes is still answering
    await asyncio.sleep(0.6)

    assert len(services.hermes_calls) == 1
    assert services.customer_texts() == ["A colleague will help."]


async def test_an_answer_that_arrives_after_a_person_took_over_becomes_a_private_note(services):
    services.hermes_delay = 0.2
    relay.enqueue(message("Hi"))
    await asyncio.sleep(0.1)
    relay.stop_answering((1, 7))  # a team member opened the conversation
    await asyncio.sleep(0.3)

    assert services.customer_texts() == []
    assert "Hello!" in services.private_notes()[0]


def test_a_status_change_away_from_pending_stops_the_bot(monkeypatch):
    stopped = []
    monkeypatch.setattr(relay, "stop_answering", stopped.append)
    body = json.dumps({"event": "conversation_status_changed", "id": 7, "status": "open", "account": {"id": 1}}).encode()

    r = TestClient(relay.app).post("/chatwoot/webhook", content=body, headers=signed(body))

    assert r.json() == {"status": "stopped"}
    assert stopped == [(1, 7)]


async def test_hermes_failure_hands_off_with_a_private_note(services):
    services.hermes_status = 500
    await relay.handle_batch([message("Hi")])

    assert "Hermes failed" in services.private_notes()[0]
    assert services.customer_texts() == [relay.HANDOFF_MESSAGE]
    assert services.sent[-1]["status"] == "open"


# ----------------------------------------------------------------------------
# Pictures from Hermes back to the customer
# ----------------------------------------------------------------------------
async def test_generated_image_is_sent_as_a_photo_after_the_text(services):
    b64 = base64.b64encode(PNG).decode()
    services.hermes_reply = f"Here is Noah's preview!\n\n![image](data:image/png;base64,{b64})"
    await relay.handle_batch([message("Show me")])

    assert services.customer_texts() == ["Here is Noah's preview!"]
    assert len(services.attachments()) == 1 and PNG in services.attachments()[0]
    assert services.sent[0].get("text") == "Here is Noah's preview!"


async def test_media_link_is_downloaded_from_the_checked_address(services):
    services.hermes_reply = "Your preview:\nMEDIA:https://cdn.example.com/p.png"
    await relay.handle_batch([message("Show me")])

    assert services.customer_texts() == ["Your preview:"]
    assert PNG in services.attachments()[0]
    assert services.fetched == [("93.184.216.34", "cdn.example.com")]


async def test_a_webp_picture_is_converted_to_jpeg_for_whatsapp(services):
    services.hermes_reply = "MEDIA:https://cdn.example.com/p.webp"
    await relay.handle_batch([message("Show me")])

    upload = services.attachments()[0]
    assert b"Content-Type: image/jpeg" in upload and b"\xff\xd8\xff" in upload


async def test_private_addresses_are_never_fetched(services):
    services.hermes_reply = "MEDIA:http://internal.example.com/secret.png"
    await relay.handle_batch([message("Show me")])

    assert services.fetched == []
    assert services.attachments() == []
    assert "not a public address" in services.private_notes()[0]


async def test_a_public_link_redirecting_to_a_private_address_is_not_followed(services):
    original = services.handler

    async def redirecting(request):
        if request.headers.get("host") == "cdn.example.com" and request.url.path == "/r.png":
            return httpx.Response(302, headers={"location": "http://internal.example.com/secret.png"})
        return await original(request)

    relay.http = httpx.AsyncClient(transport=httpx.MockTransport(redirecting))
    services.hermes_reply = "MEDIA:https://cdn.example.com/r.png"
    await relay.handle_batch([message("Show me")])

    assert [host for _, host in services.fetched] == []
    assert services.attachments() == []
    assert "not a public address" in services.private_notes()[0]


async def test_a_link_that_is_not_a_picture_stays_in_the_text_as_a_link(services):
    services.hermes_reply = "See this\nMEDIA:https://cdn.example.com/page.html"
    await relay.handle_batch([message("Show me")])

    assert services.attachments() == []
    assert services.customer_texts() == ["See this\n\nhttps://cdn.example.com/page.html"]
    assert "couldn't send" in services.private_notes()[0]


async def test_a_picture_that_cannot_be_sent_at_all_hands_off(services):
    services.hermes_reply = "MEDIA:/home/hermes/.hermes/cache/images/huge.png"
    await relay.handle_batch([message("Show me")])

    assert services.customer_texts() == [relay.HANDOFF_MESSAGE]
    assert services.sent[-1]["status"] == "open"


def test_media_tags_are_read_through_quotes_and_punctuation():
    text, sources = relay.split_media("Done! `MEDIA:https://cdn.example.com/p.png`.")
    assert text == "Done!"
    assert sources == ["https://cdn.example.com/p.png"]


def test_pictures_come_out_in_the_order_they_appear():
    text, sources = relay.split_media("MEDIA:https://a.example/1.png\n![x](https://b.example/2.png)")
    assert sources == ["https://a.example/1.png", "https://b.example/2.png"]


@pytest.mark.parametrize("prose", [
    "FOLLOW US ON SOCIAL MEDIA: @mybelovedstories",
    "MEDIA: Coming soon",
    "See MULTIMEDIA:https://example.com/x.png",
])
def test_ordinary_text_with_the_word_media_is_left_alone(prose):
    assert relay.split_media(prose) == (prose, [])


# ----------------------------------------------------------------------------
# The webhook
# ----------------------------------------------------------------------------
def signed(body: bytes, secret="hook-secret"):
    ts = str(int(time.time()))
    sig = "sha256=" + hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return {"X-Chatwoot-Timestamp": ts, "X-Chatwoot-Signature": sig, "Content-Type": "application/json"}


def test_webhook_queues_a_signed_incoming_message(monkeypatch):
    queued = []
    monkeypatch.setattr(relay, "enqueue", queued.append)
    body = json.dumps(message("Hi")).encode()

    r = TestClient(relay.app).post("/chatwoot/webhook", content=body, headers=signed(body))

    assert r.json() == {"status": "accepted"}
    assert queued[0]["content"] == "Hi"


def test_webhook_refuses_a_bad_signature(monkeypatch):
    queued = []
    monkeypatch.setattr(relay, "enqueue", queued.append)
    body = json.dumps(message("Hi")).encode()

    r = TestClient(relay.app).post("/chatwoot/webhook", content=body, headers=signed(body, secret="wrong"))

    assert r.status_code == 401 and queued == []
