"""Google sign-in for the Flask app, with an allow-list of email addresses.

Plain OAuth 2.0 code flow, no extra libraries: /login sends the browser to
Google, Google sends it back to /callback with a one-time code, and we swap the
code for an ID token over a direct HTTPS call to Google. Because that token comes
straight from Google's token endpoint (not through the browser), its signature
does not need checking (Google's own guidance); we still check audience, issuer,
expiry and that the email is verified.

Who may sign in: a JSON file (ASK_USERS_FILE), seeded once from ASK_INITIAL_ADMINS.
Admins add and remove people on /admin.

Env:
  GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET   OAuth "Web application" client
  ASK_PUBLIC_URL        e.g. https://ask.example.com (redirect URI = + /callback)
  ASK_SECRET_KEY        signs the session cookie
  ASK_USERS_FILE        default ./users.json
  ASK_INITIAL_ADMINS    comma-separated emails, used only when the file is new
"""
import base64
import functools
import html
import json
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
from datetime import timedelta

from flask import Blueprint, abort, jsonify, redirect, request, session

CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
PUBLIC_URL = os.environ.get("ASK_PUBLIC_URL", "").rstrip("/")
USERS_FILE = os.environ.get("ASK_USERS_FILE", "users.json")
INITIAL_ADMINS = [e.strip().lower() for e in
                  os.environ.get("ASK_INITIAL_ADMINS", "").split(",") if e.strip()]

bp = Blueprint("auth", __name__)
_lock = threading.Lock()
OPEN_PATHS = {"/login", "/callback", "/logout", "/healthz", "/robots.txt"}


def init_app(app):
    app.secret_key = os.environ.get("ASK_SECRET_KEY") or secrets.token_hex(32)
    app.config.update(SESSION_COOKIE_NAME="ask_session", SESSION_COOKIE_SECURE=True,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      PERMANENT_SESSION_LIFETIME=timedelta(days=30))
    app.register_blueprint(bp)
    app.before_request(_gate)


# ---------------------------------------------------------------- users
def _load():
    with _lock:
        if not os.path.exists(USERS_FILE):
            users = [{"email": e, "admin": True} for e in INITIAL_ADMINS]
            _save(users)
            return users
        with open(USERS_FILE) as f:
            return json.load(f)["users"]


def _save(users):   # caller holds _lock (or is _load during first run)
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"users": users}, f, indent=1)
    os.replace(tmp, USERS_FILE)


def find_user(email):
    email = (email or "").lower()
    return next((u for u in _load() if u["email"] == email), None)


def current_user():
    return find_user(session.get("email"))


def user_key():
    """Identity for rate limits: the signed-in email."""
    return session.get("email") or "?"


# ---------------------------------------------------------------- gate
def _gate():
    if request.path in OPEN_PATHS:
        return None
    if current_user():
        return None
    if request.path.startswith("/api/"):
        return jsonify(error="Please sign in."), 401
    return _page("Sign in",
                 "<p>This search is private. Sign in with your Google account.</p>"
                 '<p><a class=btn href="/login">Sign in with Google</a></p>'), 401


def admin_required(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        u = current_user()
        if not u or not u.get("admin"):
            abort(403)
        return fn(*a, **kw)
    return wrapper


# ---------------------------------------------------------------- OAuth
@bp.get("/login")
def login():
    if not CLIENT_ID or not PUBLIC_URL:
        return _page("Sign-in not set up", "<p>Google sign-in is not configured yet.</p>"), 503
    session["oauth_state"] = state = secrets.token_urlsafe(24)
    q = urllib.parse.urlencode({
        "client_id": CLIENT_ID, "redirect_uri": PUBLIC_URL + "/callback",
        "response_type": "code", "scope": "openid email profile",
        "state": state, "prompt": "select_account"})
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?" + q)


@bp.get("/callback")
def callback():
    state = session.pop("oauth_state", None)
    if not state or not secrets.compare_digest(state, request.args.get("state", "")):
        return _page("Sign-in failed", '<p>The sign-in link expired. <a href="/login">Try again</a>.</p>'), 400
    code = request.args.get("code")
    if not code:
        return _page("Sign-in cancelled", '<p><a href="/login">Try again</a>.</p>'), 400
    body = urllib.parse.urlencode({
        "code": code, "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
        "redirect_uri": PUBLIC_URL + "/callback", "grant_type": "authorization_code"}).encode()
    try:
        with urllib.request.urlopen("https://oauth2.googleapis.com/token", body, timeout=15) as r:
            tok = json.loads(r.read())
        payload = tok["id_token"].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception:
        return _page("Sign-in failed", '<p>Google did not confirm the sign-in. <a href="/login">Try again</a>.</p>'), 502
    if (claims.get("aud") != CLIENT_ID
            or claims.get("iss") not in ("https://accounts.google.com", "accounts.google.com")
            or claims.get("exp", 0) < time.time() or not claims.get("email_verified")):
        return _page("Sign-in failed", "<p>That sign-in could not be verified.</p>"), 403
    email = claims["email"].lower()
    if not find_user(email):
        return _page("No access",
                     f"<p>{html.escape(email)} is not on the list for this site. "
                     "Ask the owner to add you.</p>"
                     '<p><a href="/login">Use another account</a></p>'), 403
    session.clear()
    session.permanent = True
    session["email"] = email
    session["name"] = claims.get("name", "")
    session["csrf"] = secrets.token_urlsafe(16)
    return redirect("/")


@bp.get("/logout")
def logout():
    session.clear()
    return redirect("/")


# ---------------------------------------------------------------- admin
@bp.route("/admin", methods=["GET", "POST"])
@admin_required
def admin():
    msg = ""
    if request.method == "POST":
        if not secrets.compare_digest(session.get("csrf", ""), request.form.get("csrf", "")):
            abort(400)
        email = (request.form.get("email") or "").strip().lower()
        action = request.form.get("action")
        with _lock:
            with open(USERS_FILE) as f:
                users = json.load(f)["users"]
            if action == "add" and "@" in email and not any(u["email"] == email for u in users):
                users.append({"email": email, "admin": bool(request.form.get("admin"))})
                msg = f"Added {email}."
            elif action == "remove" and email != session.get("email"):
                users = [u for u in users if u["email"] != email]
                msg = f"Removed {email}."
            _save(users)
    rows = "".join(
        f"<tr><td>{html.escape(u['email'])}</td><td>{'admin' if u.get('admin') else ''}</td><td>"
        + ("" if u["email"] == session.get("email") else
           f"<form method=post><input type=hidden name=csrf value={session['csrf']}>"
           f"<input type=hidden name=email value=\"{html.escape(u['email'])}\">"
           "<button name=action value=remove>Remove</button></form>")
        + "</td></tr>" for u in _load())
    return _page("Who can sign in", (f"<p class=sub>{html.escape(msg)}</p>" if msg else "")
                 + f"<table>{rows}</table>"
                 f"<form method=post class=add><input type=hidden name=csrf value={session['csrf']}>"
                 "<input type=email name=email placeholder=email@gmail.com required> "
                 "<label><input type=checkbox name=admin> admin</label> "
                 "<button name=action value=add>Add</button></form>"
                 '<p><a href="/">Back to search</a></p>')


def _page(title, inner):
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>"
            ":root{color-scheme:light dark}body{max-width:36rem;margin:3rem auto;padding:0 1.25rem;"
            "font:1.05rem/1.6 Georgia,serif}a{color:#8a6a2f}.btn{display:inline-block;padding:.6rem 1rem;"
            "border:1px solid #8a6a2f;border-radius:4px;text-decoration:none}"
            "table{border-collapse:collapse;margin:1rem 0}td{padding:.3rem .8rem .3rem 0;border-bottom:1px solid #8884}"
            "form{display:inline}.add{display:block;margin-top:1rem}.sub{color:#777}"
            f"</style></head><body><h1>{html.escape(title)}</h1>{inner}</body></html>")
