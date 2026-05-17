from datetime import datetime, timedelta
import os
import re
import json
import random
import smtplib
from email.mime.text import MIMEText

from dotenv import load_dotenv
load_dotenv(override=True)

from googleapiclient.discovery import build
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "sentilytics-secret-key")

# ---------------- OpenAI & YouTube ----------------
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------- Email OTP ----------------
MAIL_EMAIL    = os.getenv("MAIL_EMAIL")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")

def send_otp_email(to_email: str, otp: str):
    msg = MIMEText(
        f"Your Sentilytics verification code is:\n\n"
        f"  {otp}\n\n"
        f"This code expires in 10 minutes."
    )
    msg["Subject"] = "Sentilytics — Verification Code"
    msg["From"]    = MAIL_EMAIL
    msg["To"]      = to_email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(MAIL_EMAIL, MAIL_PASSWORD)
        smtp.send_message(msg)

def generate_otp() -> str:
    return str(random.randint(100000, 999999))

# DB
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sentilytics.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------------- Models ----------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(300), nullable=False)

    is_moderator = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)

    failed_login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)

    otp_code       = db.Column(db.String(6),  nullable=True)
    otp_expires_at = db.Column(db.DateTime,   nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Search(db.Model):
    __tablename__ = "searches"
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    keyword = db.Column(db.String(200), nullable=False)
    video_url = db.Column(db.String(500), nullable=True)
    comment_limit = db.Column(db.Integer, default=30)
    comment_order = db.Column(db.String(20), default="newest")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Feedback(db.Model):
    __tablename__ = "feedbacks"
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    search_id = db.Column(db.Integer, db.ForeignKey("searches.id"), nullable=False, index=True)

    rating = db.Column(db.Integer, nullable=True)  # 1..5
    message = db.Column(db.Text, nullable=True)

    status = db.Column(db.String(20), default="NEW")  # NEW / REVIEWED
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnalysisResult(db.Model):
    __tablename__ = "analysis_results"
    id = db.Column(db.Integer, primary_key=True)

    search_id = db.Column(db.Integer, db.ForeignKey("searches.id"), nullable=False, unique=True)

    positive = db.Column(db.Integer, nullable=False)
    neutral = db.Column(db.Integer, nullable=False)
    negative = db.Column(db.Integer, nullable=False)
    labeled_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class SystemLog(db.Model):
    __tablename__ = "system_logs"
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    detail = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

# ---------------- Helpers ----------------
def is_logged_in():
    return session.get("user_id") is not None

def require_login():
    if not is_logged_in():
        return redirect(url_for("login"))
    return None

def current_user():
    if not is_logged_in():
        return None
    return db.session.get(User, session["user_id"])

def require_moderator():
    if not is_logged_in():
        return redirect(url_for("login"))
    u = current_user()
    if not u or not u.is_moderator:
        flash("Access denied.", "danger")
        return redirect(url_for("search"))
    return None

def log_event(action: str, detail: str = None):
    try:
        entry = SystemLog(
            user_id=session.get("user_id"),
            action=action,
            detail=detail,
            ip_address=request.remote_addr
        )
        db.session.add(entry)
        db.session.commit()
    except Exception:
        pass

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"https?://\S+|www\.\S+", "", s)  # remove URLs
    s = re.sub(r"[@#]", "", s)                   # remove @ and # symbols
    s = re.sub(r"\s+", " ", s).strip()           # normalize spaces
    return s

def extract_video_id(url: str) -> str:
    if not url:
        return None
    match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else None


def keyword_in_text(keyword: str, text: str) -> bool:
    pattern = r'(?<![a-zA-Z0-9])' + re.escape(keyword) + r'(?![a-zA-Z0-9])'
    return bool(re.search(pattern, text, re.IGNORECASE))


def load_comments_from_youtube(keyword: str, limit: int = 30, video_id: str = None, order: str = "newest"):
    fetch_all = limit == -1
    if not keyword or not keyword.strip() or not video_id:
        return [], 0

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        comments = []

        def parse_item(item):
            snip = item["snippet"]["topLevelComment"]["snippet"]
            raw  = snip.get("textDisplay", "")
            date = snip.get("publishedAt", "")[:10]
            return clean_text(raw), date

        fetch_target = 5000 if fetch_all else min(limit, 5000)
        yt_order = "time" if order == "newest" else "relevance"
        next_page_token = None
        fetched = 0

        while len(comments) < fetch_target and fetched < 5000:
            params = dict(part="snippet", videoId=video_id,
                          maxResults=100, textFormat="plainText", order=yt_order)
            if next_page_token:
                params["pageToken"] = next_page_token

            resp = youtube.commentThreads().list(**params).execute()
            for item in resp.get("items", []):
                text, date = parse_item(item)
                fetched += 1
                if text and len(text) > 10 and keyword_in_text(keyword, text):
                    comments.append({"text": text, "date": date})
                if len(comments) >= fetch_target:
                    break

            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break

        total = len(comments)
        if not fetch_all:
            comments = comments[:limit]

        return comments, total

    except Exception:
        return [], 0



def analyze_with_openai(comments: list) -> tuple:
    if not comments:
        return {"positive": 0, "neutral": 100, "negative": 0}, []

    BATCH_SIZE = 30
    all_labels = []
    texts = [c["text"] for c in comments]
    dates = [c.get("date", "") for c in comments]

    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            comments_text = "\n".join(f"{j+1}. {c[:300]}" for j, c in enumerate(batch))

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a sentiment analysis expert. "
                            "Label each comment as exactly one of: positive, neutral, or negative. "
                            "Return ONLY a valid JSON object in this format: "
                            "{\"labels\": [\"positive\", \"neutral\", ...]} "
                            "The array length must equal the number of input comments."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Label the sentiment of each of these {len(batch)} comments:\n\n{comments_text}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0
            )

            result = json.loads(response.choices[0].message.content)
            labels = result.get("labels", [])

            while len(labels) < len(batch):
                labels.append("neutral")
            all_labels.extend(labels[:len(batch)])

        labeled = [{"text": t, "sentiment": l, "date": d}
                   for t, d, l in zip(texts, dates, all_labels)]

        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for l in all_labels:
            if l in counts:
                counts[l] += 1

        total = sum(counts.values()) or 1
        pos = int(round(counts["positive"] / total * 100))
        neu = int(round(counts["neutral"]  / total * 100))
        neg = int(round(counts["negative"] / total * 100))
        neu += 100 - (pos + neu + neg)

        return {"positive": pos, "neutral": neu, "negative": neg}, labeled

    except Exception:
        labeled = [{"text": c["text"], "sentiment": "neutral", "date": c.get("date", "")}
                   for c in comments]
        return {"positive": 0, "neutral": 100, "negative": 0}, labeled


# ---------------- Routes ----------------
@app.get("/")
def root():
    if is_logged_in():
        return redirect(url_for("search"))
    return redirect(url_for("login"))

# ---------- AUTH ----------
@app.get("/login")
def login():
    return render_template("auth_login.html")

@app.post("/login")
def login_post():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()

    user = User.query.filter_by(email=email).first()

    if not user:
        flash("Invalid email or password.", "danger")
        return redirect(url_for("login"))

    # 🔒 Security: if locked, block login
    if user.locked_until and user.locked_until > datetime.utcnow():
        remaining_minutes = int((user.locked_until - datetime.utcnow()).total_seconds() / 60)
        if remaining_minutes < 1:
            remaining_minutes = 1
        flash(f"Account locked. Try again in {remaining_minutes} minutes.", "danger")
        return redirect(url_for("login"))

    # Check password
    if not check_password_hash(user.password_hash, password):
        user.failed_login_attempts += 1

        # Lock account after 5 failed attempts
        if user.failed_login_attempts >= 5:
            user.locked_until = datetime.utcnow() + timedelta(minutes=10)
            user.failed_login_attempts = 0
            flash("Too many failed attempts. Account locked for 10 minutes.", "danger")
        else:
            flash("Invalid email or password.", "danger")

        db.session.commit()
        return redirect(url_for("login"))

    # ✅ Successful login: reset counters
    user.failed_login_attempts = 0
    user.locked_until = None
    db.session.commit()

    # prevent disabled accounts
    if not user.is_active:
        flash("Your account is disabled. Please contact the moderator at sentilyticss@gmail.com.", "danger")
        return redirect(url_for("login"))

    # Send OTP
    otp = generate_otp()
    user.otp_code       = otp
    user.otp_expires_at = datetime.utcnow() + timedelta(minutes=10)
    db.session.commit()

    try:
        send_otp_email(user.email, otp)
    except Exception:
        flash("Failed to send OTP email. Try again.", "danger")
        return redirect(url_for("login"))

    session["otp_user_id"] = user.id
    flash("A verification code was sent to your email.", "info")
    return redirect(url_for("verify_login_otp"))

@app.get("/register")
def register():
    return render_template("auth_register.html")

@app.post("/register")
def register_post():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()

    if not name or not email or not password:
        flash("Please fill all fields.", "warning")
        return redirect(url_for("register"))

    if User.query.filter_by(email=email).first():
        flash("This email is already registered. Please login.", "warning")
        return redirect(url_for("login"))

    # Password strength validation
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "warning")
        return redirect(url_for("register"))
    if not re.search(r"[A-Z]", password):
        flash("Password must contain at least one uppercase letter.", "warning")
        return redirect(url_for("register"))
    if not re.search(r"[0-9]", password):
        flash("Password must contain at least one number.", "warning")
        return redirect(url_for("register"))
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        flash("Password must contain at least one special character.", "warning")
        return redirect(url_for("register"))

    otp = generate_otp()
    session["pending_register"] = {
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "otp": otp,
        "otp_expires_at": (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    }

    try:
        send_otp_email(email, otp)
    except Exception:
        flash("Failed to send OTP email. Try again.", "danger")
        return redirect(url_for("register"))

    flash("A verification code was sent to your email.", "info")
    return redirect(url_for("verify_register_otp"))

@app.get("/verify-login")
def verify_login_otp():
    if "otp_user_id" not in session:
        return redirect(url_for("login"))
    return render_template("verify_otp.html", mode="login")

@app.post("/verify-login")
def verify_login_otp_post():
    if "otp_user_id" not in session:
        return redirect(url_for("login"))

    user = db.session.get(User, session["otp_user_id"])
    if not user:
        return redirect(url_for("login"))

    entered = request.form.get("otp", "").strip()

    if datetime.utcnow() > user.otp_expires_at:
        flash("Code expired. Please login again.", "danger")
        session.pop("otp_user_id", None)
        return redirect(url_for("login"))

    if entered != user.otp_code:
        flash("Invalid code. Try again.", "danger")
        return redirect(url_for("verify_login_otp"))

    user.otp_code = None
    user.otp_expires_at = None
    db.session.commit()

    session.pop("otp_user_id", None)
    session["user_id"] = user.id
    session["user_name"] = user.name
    session["is_moderator"] = user.is_moderator
    log_event("LOGIN", f"User logged in: {user.email}")
    return redirect(url_for("search"))


@app.get("/verify-register")
def verify_register_otp():
    if "pending_register" not in session:
        return redirect(url_for("register"))
    return render_template("verify_otp.html", mode="register")

@app.post("/verify-register")
def verify_register_otp_post():
    pending = session.get("pending_register")
    if not pending:
        return redirect(url_for("register"))

    entered = request.form.get("otp", "").strip()
    expires = datetime.fromisoformat(pending["otp_expires_at"])

    if datetime.utcnow() > expires:
        session.pop("pending_register", None)
        flash("Code expired. Please register again.", "danger")
        return redirect(url_for("register"))

    if entered != pending["otp"]:
        flash("Invalid code. Try again.", "danger")
        return redirect(url_for("verify_register_otp"))

    user = User(
        name=pending["name"],
        email=pending["email"],
        password_hash=pending["password_hash"]
    )
    db.session.add(user)
    db.session.commit()

    session.pop("pending_register", None)
    log_event("REGISTER", f"New account: {user.email}")
    flash("Account created successfully. Please login.", "success")
    return redirect(url_for("login"))


@app.get("/logout")
def logout():
    log_event("LOGOUT")
    session.clear()
    return redirect(url_for("login"))

# ---------- FORGOT PASSWORD ----------
@app.get("/forgot-password")
def forgot_password():
    return render_template("forgot_password.html", step="email")

@app.post("/forgot-password")
def forgot_password_post():
    email = request.form.get("email", "").strip().lower()
    user = User.query.filter_by(email=email).first()
    if user and user.is_active:
        otp = generate_otp()
        user.otp_code = otp
        user.otp_expires_at = datetime.utcnow() + timedelta(minutes=10)
        db.session.commit()
        try:
            send_otp_email(user.email, otp)
        except Exception:
            flash("Failed to send reset code. Try again.", "danger")
            return redirect(url_for("forgot_password"))
        session["reset_user_id"] = user.id
    # Don't reveal whether the email exists
    flash("If this email is registered, a reset code has been sent.", "info")
    return redirect(url_for("forgot_password_verify"))

@app.get("/forgot-password/verify")
def forgot_password_verify():
    if "reset_user_id" not in session:
        return redirect(url_for("forgot_password"))
    return render_template("forgot_password.html", step="verify")

@app.post("/forgot-password/verify")
def forgot_password_verify_post():
    if "reset_user_id" not in session:
        return redirect(url_for("forgot_password"))

    user = db.session.get(User, session["reset_user_id"])
    if not user:
        session.pop("reset_user_id", None)
        return redirect(url_for("forgot_password"))

    entered = request.form.get("otp", "").strip()

    if not user.otp_expires_at or datetime.utcnow() > user.otp_expires_at:
        flash("Code expired. Please request a new one.", "danger")
        session.pop("reset_user_id", None)
        return redirect(url_for("forgot_password"))

    if entered != user.otp_code:
        flash("Invalid code. Try again.", "danger")
        return redirect(url_for("forgot_password_verify"))

    user.otp_code = None
    user.otp_expires_at = None
    db.session.commit()

    session["reset_verified"] = True
    return redirect(url_for("forgot_password_reset"))

@app.get("/forgot-password/reset")
def forgot_password_reset():
    if not session.get("reset_verified") or "reset_user_id" not in session:
        return redirect(url_for("forgot_password"))
    return render_template("forgot_password.html", step="reset")

@app.post("/forgot-password/reset")
def forgot_password_reset_post():
    if not session.get("reset_verified") or "reset_user_id" not in session:
        return redirect(url_for("forgot_password"))

    user = db.session.get(User, session["reset_user_id"])
    if not user:
        session.pop("reset_user_id", None)
        session.pop("reset_verified", None)
        return redirect(url_for("forgot_password"))

    new_pw = request.form.get("new_password", "").strip()
    confirm_pw = request.form.get("confirm_password", "").strip()

    if len(new_pw) < 8:
        flash("Password must be at least 8 characters.", "warning")
        return redirect(url_for("forgot_password_reset"))
    if not re.search(r"[A-Z]", new_pw):
        flash("Password must contain at least one uppercase letter.", "warning")
        return redirect(url_for("forgot_password_reset"))
    if not re.search(r"[0-9]", new_pw):
        flash("Password must contain at least one number.", "warning")
        return redirect(url_for("forgot_password_reset"))
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", new_pw):
        flash("Password must contain at least one special character.", "warning")
        return redirect(url_for("forgot_password_reset"))
    if new_pw != confirm_pw:
        flash("Passwords do not match.", "warning")
        return redirect(url_for("forgot_password_reset"))

    user.password_hash = generate_password_hash(new_pw)
    user.failed_login_attempts = 0
    user.locked_until = None
    db.session.commit()

    session.pop("reset_user_id", None)
    session.pop("reset_verified", None)

    log_event("PASSWORD_RESET", f"Password reset via forgot-password: {user.email}")
    flash("Password reset successfully. You can now sign in.", "success")
    return redirect(url_for("login"))

# ---------- SEARCH ----------
@app.get("/search")
def search():
    guard = require_login()
    if guard:
        return guard

    searches = (
        Search.query
        .filter_by(user_id=session["user_id"])
        .order_by(Search.created_at.desc())
        .limit(5)
        .all()
    )
    return render_template("search.html", searches=searches)

@app.post("/search")
def search_post():
    guard = require_login()
    if guard:
        return guard

    keyword = request.form.get("keyword", "").strip()
    video_url = request.form.get("video_url", "").strip()

    if request.form.get("fetch_all") == "1":
        comment_limit = -1
    else:
        try:
            comment_limit = int(request.form.get("comment_limit", 30))
        except ValueError:
            comment_limit = 30
        comment_limit = max(1, min(5000, comment_limit))

    comment_order = request.form.get("comment_order", "newest")
    if comment_order not in ("newest", "relevant"):
        comment_order = "newest"

    if not keyword:
        flash("Please enter a keyword.", "warning")
        return redirect(url_for("search"))

    if not video_url:
        flash("Please enter a YouTube video URL.", "warning")
        return redirect(url_for("search"))

    if not extract_video_id(video_url):
        flash("Invalid YouTube URL. Please paste a valid video link.", "warning")
        return redirect(url_for("search"))

    existing_search = Search.query.filter_by(
        user_id=session["user_id"],
        keyword=keyword,
        video_url=video_url,
        comment_limit=comment_limit,
        comment_order=comment_order
    ).first()

    if existing_search:
        existing_search.created_at = datetime.utcnow()
        db.session.commit()
        search_id = existing_search.id
    else:
        s = Search(
            user_id=session["user_id"],
            keyword=keyword,
            video_url=video_url,
            comment_limit=comment_limit,
            comment_order=comment_order
        )
        db.session.add(s)
        db.session.commit()
        search_id = s.id

    # Keep only last 5 searches — delete older ones
    all_searches = (
        Search.query
        .filter_by(user_id=session["user_id"])
        .order_by(Search.created_at.desc())
        .all()
    )
    for old in all_searches[5:]:
        AnalysisResult.query.filter_by(search_id=old.id).delete(synchronize_session=False)
        Feedback.query.filter_by(search_id=old.id).delete(synchronize_session=False)
        db.session.delete(old)
    db.session.commit()

    log_event("SEARCH", f"Keyword: {keyword} | Video: {video_url} | Limit: {comment_limit} | Order: {comment_order}")
    return redirect(url_for("dashboard", search_id=search_id))

@app.post("/search/delete/<int:search_id>")
def delete_search(search_id):
    guard = require_login()
    if guard:
        return guard

    s = Search.query.filter_by(id=search_id, user_id=session["user_id"]).first()
    if s:
        AnalysisResult.query.filter_by(search_id=s.id).delete(synchronize_session=False)
        Feedback.query.filter_by(search_id=s.id).delete(synchronize_session=False)
        db.session.delete(s)
        db.session.commit()

    return redirect(url_for("search"))


@app.get("/search/open/<int:search_id>")
def open_recent_search(search_id):
    guard = require_login()
    if guard:
        return guard

    s = Search.query.filter_by(id=search_id, user_id=session["user_id"]).first()
    if not s:
        flash("Search not found.", "danger")
        return redirect(url_for("search"))

    s.created_at = datetime.utcnow()
    db.session.commit()
    return redirect(url_for("dashboard", search_id=s.id))

# ---------- DASHBOARD (Latest) ----------
@app.get("/dashboard")
def dashboard_latest():
    guard = require_login()
    if guard:
        return guard

    last_search = (
        Search.query
        .filter_by(user_id=session["user_id"])
        .order_by(Search.created_at.desc())
        .first()
    )

    if not last_search:
        flash("No searches yet. Please perform a search first.", "warning")
        return redirect(url_for("search"))

    return redirect(url_for("dashboard", search_id=last_search.id))

# ---------- DASHBOARD (By search id) ----------
@app.get("/dashboard/<int:search_id>")
def dashboard(search_id):
    guard = require_login()
    if guard:
        return guard

    s = Search.query.filter_by(id=search_id, user_id=session["user_id"]).first()
    if not s:
        flash("Search not found.", "danger")
        return redirect(url_for("search"))

    analysis = AnalysisResult.query.filter_by(search_id=search_id).first()

    if analysis and analysis.labeled_json:
        stats = {"positive": analysis.positive, "neutral": analysis.neutral, "negative": analysis.negative}
        labeled_comments = json.loads(analysis.labeled_json)
        total_matches = len(labeled_comments)
    else:
        video_id = extract_video_id(s.video_url)
        limit = s.comment_limit or 30
        order = s.comment_order or "newest"
        comments, total_matches = load_comments_from_youtube(s.keyword, limit=limit, video_id=video_id, order=order)

        if comments:
            stats, labeled_comments = analyze_with_openai(comments)
        else:
            stats = {"positive": 0, "neutral": 100, "negative": 0}
            labeled_comments = []

        labeled_json = json.dumps(labeled_comments, ensure_ascii=False)
        if analysis:
            analysis.positive     = stats["positive"]
            analysis.neutral      = stats["neutral"]
            analysis.negative     = stats["negative"]
            analysis.labeled_json = labeled_json
        else:
            db.session.add(AnalysisResult(
                search_id=search_id,
                positive=stats["positive"],
                neutral=stats["neutral"],
                negative=stats["negative"],
                labeled_json=labeled_json
            ))
        db.session.commit()

    return render_template(
        "dashboard.html",
        keyword=s.keyword,
        total_matches=total_matches,
        labeled_comments=labeled_comments,
        stats=stats,
        search_id=search_id
    )


# ---------- FEEDBACK ----------
@app.post("/feedback/<int:search_id>")
def submit_feedback(search_id):
    guard = require_login()
    if guard:
        return guard

    s = Search.query.filter_by(id=search_id, user_id=session["user_id"]).first()
    if not s:
        flash("Search not found.", "danger")
        return redirect(url_for("search"))

    rating_raw = request.form.get("rating", "").strip()
    message = request.form.get("message", "").strip()

    rating = None
    if rating_raw.isdigit():
        rating = int(rating_raw)
        if rating < 1 or rating > 5:
            rating = None

    fb = Feedback(
        user_id=session["user_id"],
        search_id=search_id,
        rating=rating,
        message=message
    )
    db.session.add(fb)
    db.session.commit()

    log_event("FEEDBACK", f"Search #{search_id} — rating: {rating}")
    flash("Thanks! Your feedback was sent.", "success")
    return redirect(url_for("dashboard", search_id=search_id))

# ---------- MODERATOR: USERS ----------
@app.get("/moderator/users")
def moderator_users():
    guard = require_moderator()
    if guard:
        return guard

    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("moderator_users.html", users=users)

@app.post("/moderator/users/<int:user_id>/toggle")
def moderator_toggle_user(user_id):
    guard = require_moderator()
    if guard:
        return guard

    u = db.session.get(User, user_id)
    if not u:
        flash("User not found.", "danger")
        return redirect(url_for("moderator_users"))

    if u.id == session["user_id"]:
        flash("You cannot disable your own account.", "warning")
        return redirect(url_for("moderator_users"))

    if u.is_moderator:
        flash("You cannot disable another moderator's account.", "warning")
        return redirect(url_for("moderator_users"))

    u.is_active = not u.is_active
    db.session.commit()

    status_str = "disabled" if not u.is_active else "enabled"
    log_event("MOD_TOGGLE_USER", f"User #{u.id} ({u.email}) {status_str}")
    flash("User status updated.", "success")
    return redirect(url_for("moderator_users"))

@app.post("/moderator/users/<int:user_id>/delete")
def moderator_delete_user(user_id):
    guard = require_moderator()
    if guard:
        return guard

    u = db.session.get(User, user_id)
    if not u:
        flash("User not found.", "danger")
        return redirect(url_for("moderator_users"))

    if u.id == session["user_id"]:
        flash("You cannot delete your own account.", "warning")
        return redirect(url_for("moderator_users"))

    if u.is_moderator:
        flash("You cannot delete another moderator's account.", "warning")
        return redirect(url_for("moderator_users"))

    Feedback.query.filter_by(user_id=u.id).delete(synchronize_session=False)

    search_ids = [s.id for s in Search.query.filter_by(user_id=u.id).all()]
    if search_ids:
        AnalysisResult.query.filter(AnalysisResult.search_id.in_(search_ids)).delete(synchronize_session=False)
        Feedback.query.filter(Feedback.search_id.in_(search_ids)).delete(synchronize_session=False)
        Search.query.filter_by(user_id=u.id).delete(synchronize_session=False)

    log_event("MOD_DELETE_USER", f"Deleted user #{u.id} ({u.email})")
    db.session.delete(u)
    db.session.commit()

    flash("User deleted successfully.", "success")
    return redirect(url_for("moderator_users"))

# ---------- MODERATOR: FEEDBACKS ----------
@app.get("/moderator/feedbacks")
def moderator_feedbacks():
    guard = require_moderator()
    if guard:
        return guard

    items = (
        db.session.query(Feedback, User, Search)
        .join(User, Feedback.user_id == User.id)
        .join(Search, Feedback.search_id == Search.id)
        .order_by(Feedback.created_at.desc())
        .all()
    )
    return render_template("moderator_feedbacks.html", items=items)

@app.post("/moderator/feedbacks/<int:feedback_id>/review")
def mark_feedback_reviewed(feedback_id):
    guard = require_moderator()
    if guard:
        return guard

    fb = db.session.get(Feedback, feedback_id)
    if not fb:
        flash("Feedback not found.", "danger")
        return redirect(url_for("moderator_feedbacks"))

    fb.status = "REVIEWED"
    db.session.commit()

    log_event("MOD_REVIEW_FEEDBACK", f"Feedback #{feedback_id} marked reviewed")
    flash("Marked as reviewed.", "success")
    return redirect(url_for("moderator_feedbacks"))


# ---------- ACCOUNT: EDIT PROFILE ----------
@app.get("/account/edit")
def account_edit():
    guard = require_login()
    if guard:
        return guard
    u = current_user()
    return render_template("account_edit.html", user=u)

@app.post("/account/edit")
def account_edit_post():
    guard = require_login()
    if guard:
        return guard

    u = current_user()
    name = request.form.get("name", "").strip()

    if not name:
        flash("Name is required.", "warning")
        return redirect(url_for("account_edit"))

    old_name = u.name
    u.name = name
    db.session.commit()

    session["user_name"] = name
    log_event("ACCOUNT_EDIT", f"Name: {old_name} → {name}")
    flash("Profile updated successfully.", "success")
    return redirect(url_for("account_edit"))


# ---------- ACCOUNT: CHANGE PASSWORD ----------
@app.get("/account/password")
def account_password():
    guard = require_login()
    if guard:
        return guard
    return render_template("account_password.html")

@app.post("/account/password")
def account_password_post():
    guard = require_login()
    if guard:
        return guard

    u = current_user()
    current_pw = request.form.get("current_password", "").strip()
    new_pw = request.form.get("new_password", "").strip()
    confirm_pw = request.form.get("confirm_password", "").strip()

    if not check_password_hash(u.password_hash, current_pw):
        flash("Current password is incorrect.", "danger")
        return redirect(url_for("account_password"))

    if len(new_pw) < 8:
        flash("Password must be at least 8 characters.", "warning")
        return redirect(url_for("account_password"))
    if not re.search(r"[A-Z]", new_pw):
        flash("Password must contain at least one uppercase letter.", "warning")
        return redirect(url_for("account_password"))
    if not re.search(r"[0-9]", new_pw):
        flash("Password must contain at least one number.", "warning")
        return redirect(url_for("account_password"))
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", new_pw):
        flash("Password must contain at least one special character.", "warning")
        return redirect(url_for("account_password"))
    if new_pw != confirm_pw:
        flash("New passwords do not match.", "warning")
        return redirect(url_for("account_password"))

    u.password_hash = generate_password_hash(new_pw)
    db.session.commit()

    log_event("PASSWORD_CHANGE", "User changed their password")
    flash("Password changed successfully.", "success")
    return redirect(url_for("account_password"))


# ---------- ACCOUNT: DELETE ----------
@app.post("/account/delete")
def account_delete():
    guard = require_login()
    if guard:
        return guard

    u = current_user()
    password = request.form.get("password", "").strip()

    if not check_password_hash(u.password_hash, password):
        flash("Incorrect password. Account not deleted.", "danger")
        return redirect(url_for("account_edit"))

    log_event("ACCOUNT_DELETE", f"User deleted own account: {u.email}")

    Feedback.query.filter_by(user_id=u.id).delete(synchronize_session=False)
    search_ids = [s.id for s in Search.query.filter_by(user_id=u.id).all()]
    if search_ids:
        Feedback.query.filter(Feedback.search_id.in_(search_ids)).delete(synchronize_session=False)
        AnalysisResult.query.filter(AnalysisResult.search_id.in_(search_ids)).delete(synchronize_session=False)
        Search.query.filter_by(user_id=u.id).delete(synchronize_session=False)

    db.session.delete(u)
    db.session.commit()

    session.clear()
    flash("Your account has been deleted.", "info")
    return redirect(url_for("login"))


# ---------- MODERATOR: LOGS ----------
@app.get("/moderator/logs")
def moderator_logs():
    guard = require_moderator()
    if guard:
        return guard

    logs = (
        db.session.query(SystemLog, User)
        .outerjoin(User, SystemLog.user_id == User.id)
        .order_by(SystemLog.created_at.desc())
        .limit(200)
        .all()
    )
    return render_template("moderator_logs.html", logs=logs)


if __name__ == "__main__":
    app.run(debug=True)