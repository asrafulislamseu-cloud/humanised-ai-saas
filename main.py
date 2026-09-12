import os
import json
import asyncio
import time
import random
import itertools
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Depends, HTTPException, Response, Cookie
from typing import Optional, Dict
from enum import Enum
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ডাটাবেজ ইম্পোর্ট (SQLAlchemy)
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

load_dotenv()

# --- 1. ডাটাবেজ সেটআপ ---
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_saas_platform.db")

if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    user_api_key = Column(String, nullable=False)     # ইউজারের নিজস্ব জেমিনি কি
    is_premium = Column(Boolean, default=False)
    message_limit = Column(Integer, default=0)       # মোট মাস্টার কি ব্যবহারের কোটা
    messages_used = Column(Integer, default=0)       # মাস্টার কি থেকে কতগুলো মেসেজ খরচ হলো
    expiry_date = Column(DateTime, nullable=True)    # ৩০ দিনের মেয়াদ
    professional_bio = Column(Text, nullable=True)   # ইন্টারভিউ বা প্রফেশনাল মোডের বায়ো

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 2. মাল্টি-এপিআই কি ও মাস্টার কি পুল সেটআপ ---
raw_user_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
if raw_user_keys:
    API_KEYS = [k.strip() for k in raw_user_keys.split(",") if k.strip()]
else:
    API_KEYS = []

raw_master_keys = os.getenv("GEMINI_MASTER_KEYS", "")
MASTER_API_KEYS = [k.strip() for k in raw_master_keys.split(",") if k.strip()]
master_key_cycle = itertools.cycle(MASTER_API_KEYS) if MASTER_API_KEYS else None

# সার্ভার ক্র্যাশ রোধে কনকারেন্সি কন্ট্রোল সেমাফোর
MAX_CONCURRENT_AI_CALLS = 20
ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_CALLS)

# রেন্ডার সার্ভারের জন্য app অবজেক্ট ইনিশিয়ালাইজেশন
app = FastAPI(title="Humanised AI SaaS Platform with Flexible Top-up Logic", version="11.0")

AUDIO_UPLOAD_DIR = "uploaded_voices"
os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)
VOICE_DB_FILE = "voice_settings.json"

def load_voice_settings():
    if os.path.exists(VOICE_DB_FILE):
        with open(VOICE_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "Mother": "default_mother_voice",
        "Father": "default_father_voice",
        "Wife": "default_wife_voice",
        "Girlfriend": "default_gf_voice",
        "Friend": "default_friend_voice",
        "User_Own_Voice": "default_user_presentation_voice"
    }

def save_voice_settings(data):
    with open(VOICE_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

USER_VOICE_SETTINGS = load_voice_settings()

class ModeEnum(str, Enum):
    emotional_chat = "emotional_chat"
    presentation = "presentation"

class PersonaEnum(str, Enum):
    Mother = "Mother"
    Father = "Father"
    Wife = "Wife"
    Girlfriend = "Girlfriend"
    Friend = "Friend"
    Professional = "Professional"
    Mentor = "Mentor"

class LanguageEnum(str, Enum):
    Bengali = "bn"
    English = "en"
    Hindi = "hi"

@app.get("/")
def read_root():
    return {"message": "AI Platform is running with Flexible Top-up & Smart Fallback Logic!"}

# --- 3. স্মার্ট এআই কল ---
async def call_gemini_with_smart_fallback(user, contents, temp_val, max_tokens, is_stream=False):
    keys_to_try = []
    
    # ১. সব সময় সবার আগে ইউজারের নিজের কি দিয়ে চেষ্টা করা হবে
    keys_to_try.append(("user", user.user_api_key))
    
    # ২. ইউজার প্রিমিয়াম হলে এবং তার মাস্টার কি কোটা বাকি থাকলে ফলব্যাক হিসেবে মাস্টার কি যুক্ত হবে
    if user.is_premium and MASTER_API_KEYS and user.messages_used < user.message_limit:
        for _ in range(min(3, len(MASTER_API_KEYS))):
            keys_to_try.append(("master", next(master_key_cycle)))

    last_exception = None
    
    for key_type, api_key in keys_to_try:
        if key_type == "master" and (not user.is_premium or user.messages_used >= user.message_limit):
            continue

        for attempt in range(2):
            try:
                async with ai_semaphore:
                    client = genai.Client(api_key=api_key)
                    if is_stream:
                        response_stream = client.models.generate_content_stream(
                            model="gemini-3.6-flash",
                            contents=contents,
                            config=types.GenerateContentConfig(temperature=temp_val, max_output_tokens=max_tokens)
                        )
                        return response_stream, key_type
                    else:
                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=contents,
                            config=types.GenerateContentConfig(temperature=temp_val, max_output_tokens=max_tokens)
                        )
                        return response.text.strip(), key_type
            except Exception as e:
                error_str = str(e)
                last_exception = e
                
                if key_type == "user":
                    if not user.is_premium:
                        break
                    if any(err in error_str for err in ["429", "ResourceExhausted", "Quota", "503", "ServiceUnavailable"]):
                        break 
                
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                else:
                    break

    raise last_exception

# --- 4. ইউজার রেজিস্ট্রেশন বা লগইন রাউটার ---
class UserRegisterRequest(BaseModel):
    email: str
    user_api_key: str

@app.post("/register-or-login")
def register_or_login(data: UserRegisterRequest, response: Response, db: Session = Depends(get_db)):
    if not data.user_api_key or not data.user_api_key.strip():
        raise HTTPException(status_code=400, detail="Please provide your Gemini API key.")
    
    user = db.query(UserDB).filter(UserDB.email == data.email).first()
    if not user:
        user = UserDB(email=data.email, user_api_key=data.user_api_key.strip(), is_premium=False, message_limit=0)
        db.add(user)
    else:
        user.user_api_key = data.user_api_key.strip()
    
    db.commit()
    
    response.set_cookie(key="current_user_email", value=data.email, httponly=True)
    return {"status": "success", "message": "Successfully logged in and active session created!"}

class UserProfileUpdate(BaseModel):
    professional_bio: Optional[str] = None

@app.post("/update-profile")
def update_user_profile(
    data: UserProfileUpdate, 
    current_user_email: Optional[str] = Cookie(None), 
    db: Session = Depends(get_db)
):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Not logged in. Please register or login first.")
        
    user = db.query(UserDB).filter(UserDB.email == current_user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    if data.professional_bio is not None:
        user.professional_bio = data.professional_bio
        db.commit()
        
    return {"status": "success", "message": "Professional profile/bio updated successfully!"}

# --- 5. ফ্লেক্সিবল সাবস্ক্রিপশন ও টপ-আপ ---
class SubscriptionActivateRequest(BaseModel):
    package_type: str

@app.post("/activate-subscription")
def activate_subscription(
    data: SubscriptionActivateRequest, 
    current_user_email: Optional[str] = Cookie(None), 
    db: Session = Depends(get_db)
):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Not logged in. Please register or login first.")
        
    user = db.query(UserDB).filter(UserDB.email == current_user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Please log in or create an account first.")
        
    added_limit = 0
    if data.package_type == "1_dollar":
        added_limit = 2500
    elif data.package_type == "4_dollar":
        added_limit = 12000
    else:
        raise HTTPException(status_code=400, detail="Invalid package type!")
        
    now = datetime.utcnow()
    
    if user.is_premium and user.expiry_date and user.expiry_date > now:
        user.message_limit += added_limit
        user.expiry_date = user.expiry_date + timedelta(days=30)
    else:
        user.message_limit = added_limit
        user.messages_used = 0 
        user.expiry_date = now + timedelta(days=30)
        
    user.is_premium = True
    db.commit()
    db.refresh(user)
    
    return {
        "status": "success",
        "message": "Subscription/Top-up added successfully!",
        "total_message_limit": user.message_limit,
        "messages_used": user.messages_used,
        "remaining_messages": max(0, user.message_limit - user.messages_used),
        "expiry_date": user.expiry_date
    }

@app.post("/upload-persona-voice")
async def upload_persona_voice(
    persona_or_mode: PersonaEnum = Form(...),
    voice_file: UploadFile = File(...)
):
    try:
        persona_val = persona_or_mode.value
        file_extension = os.path.splitext(voice_file.filename)[1]
        safe_name = persona_val.replace(" ", "_").replace("(", "").replace(")", "")
        saved_filename = f"{safe_name}_voice{file_extension}"
        file_path = os.path.join(AUDIO_UPLOAD_DIR, saved_filename)
        
        contents = await voice_file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            
        USER_VOICE_SETTINGS[persona_val] = file_path
        save_voice_settings(USER_VOICE_SETTINGS)
        
        return {"status": "success", "voice_file_path": file_path}
    except Exception as e:
        return {"error": str(e)}

# --- 6. মূল এআই প্রসেসিং রাউটার ---
@app.post("/process-ai")
async def process_ai_request(
    mode: ModeEnum = Form(...),
    persona: Optional[PersonaEnum] = Form(PersonaEnum.Mother),
    target_language: Optional[LanguageEnum] = Form(LanguageEnum.Bengali),
    user_message: Optional[str] = Form(""),
    slide_content: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
    current_user_email: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
):
    try:
        if not current_user_email:
            raise HTTPException(status_code=401, detail="Not logged in. Please use /register-or-login first.")
            
        user = db.query(UserDB).filter(UserDB.email == current_user_email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
            
        if user.is_premium and user.expiry_date and datetime.utcnow() > user.expiry_date:
            user.is_premium = False
            db.commit()

        contents = []
        if file:
            file_bytes = await file.read()
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type=file.content_type))

        mode_val = mode.value
        persona_val = persona.value if persona else "Mother"
        lang_val = target_language.value if target_language else "bn"
        
        if mode_val == "presentation":
            assigned_voice_file = USER_VOICE_SETTINGS.get("User_Own_Voice", "default_user_voice")
            user_bio = user.professional_bio if user.professional_bio else "No specific candidate bio provided."
            
            prompt = (
                f"You are attending a professional job/viva interview as the candidate. "
                f"Candidate's Personal Profile & Background: {user_bio}. "
                f"Language: {lang_val}. "
                f"Topic/Context: {slide_content}. "
                f"Question asked by interviewer: {user_message}. "
                f"Instructions: Give a confident, natural, and professional answer reflecting the candidate's background."
            )
            max_tokens = 1500
            temp_val = 0.3
        else:
            assigned_voice_file = USER_VOICE_SETTINGS.get(persona_val, "default_voice")
            prompt = f"Act warmly and naturally as: {persona_val}. Language: {lang_val}. Message: {user_message}"
            max_tokens = 1500
            temp_val = 0.5

        contents.append(prompt)
        
        ai_response_text, key_used = await call_gemini_with_smart_fallback(user, contents, temp_val, max_tokens, is_stream=False)
        
        if key_used == "master":
            user.messages_used += 1
            db.commit()
        
        remaining = max(0, user.message_limit - user.messages_used) if user.is_premium else 0

        return {
            "status": "success",
            "active_user": current_user_email,
            "key_used": key_used,
            "assigned_voice_file": assigned_voice_file,
            "response": ai_response_text,
            "remaining_messages": remaining
        }
            
    except Exception as e:
        error_msg = str(e)
        if any(err in error_msg for err in ["429", "ResourceExhausted", "Quota"]):
            if not user.is_premium:
                return {"error": "Rate limit exceeded! Please wait a minute or buy a top-up package."}
            else:
                return {"error": "Rate limit exceeded and master key quota is exhausted! Please top-up more messages or wait a minute."}
        return {"error": error_msg}

# --- 7. ওয়েবসকেট এন্ডপয়েন্ট (রিয়েল-টাইম স্ট্রিম) ---
active_tasks: Dict[WebSocket, asyncio.Task] = {}

async def handle_ai_stream(websocket: WebSocket, data: dict, db: Session, current_user_email: Optional[str]):
    try:
        if not current_user_email:
            await websocket.send_json({"status": "error", "message": "Not logged in."})
            return

        user = db.query(UserDB).filter(UserDB.email == current_user_email).first()
        if not user:
            await websocket.send_json({"status": "error", "message": "User not found."})
            return

        if user.is_premium and user.expiry_date and datetime.utcnow() > user.expiry_date:
            user.is_premium = False
            db.commit()

        mode = data.get("mode", "emotional_chat")
        persona = data.get("persona", "Mother")
        target_language = data.get("target_language", "bn")
        user_message = data.get("user_message", "")
        
        if mode == "presentation":
            user_bio = user.professional_bio if user.professional_bio else "No specific candidate bio provided."
            prompt = f"You are attending a professional interview as the candidate. Profile: {user_bio}. Language: {target_language}. Question: {user_message}"
        else:
            prompt = f"Act as {persona} in language {target_language}. Message: {user_message}"

        await websocket.send_json({"status": "started"})

        response_stream, key_used = await call_gemini_with_smart_fallback(user, [prompt], temp_val=0.5, max_tokens=1500, is_stream=True)
        
        for chunk in response_stream:
            if chunk.text:
                await websocket.send_json({"status": "streaming", "chunk": chunk.text})
            await asyncio.sleep(0.0001)
        
        if key_used == "master":
            user.messages_used += 1
            db.commit()

        remaining = max(0, user.message_limit - user.messages_used) if user.is_premium else 0
        await websocket.send_json({"status": "completed", "key_used": key_used, "remaining_messages": remaining})
        
    except asyncio.CancelledError:
        await websocket.send_json({"status": "interrupted"})
        raise
    except Exception as e:
        error_msg = str(e)
        if any(err in error_msg for err in ["429", "ResourceExhausted", "Quota"]):
            await websocket.send_json({"status": "error", "message": "Rate limit exceeded! Please top-up or wait a minute."})
        else:
            await websocket.send_json({"status": "error", "message": error_msg})

@app.websocket("/ws/live-ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    cookies = websocket.cookies
    current_user_email = cookies.get("current_user_email")
    
    db = SessionLocal()
    try:
        while True:
            data = await websocket.receive_json()
            if websocket in active_tasks and not active_tasks[websocket].done():
                active_tasks[websocket].cancel()
                try:
                    await active_tasks[websocket]
                except asyncio.CancelledError:
                    pass

            task = asyncio.create_task(handle_ai_stream(websocket, data, db, current_user_email))
            active_tasks[websocket] = task
    except WebSocketDisconnect:
        if websocket in active_tasks and not active_tasks[websocket].done():
            active_tasks[websocket].cancel()
        print("WebSocket disconnected.")
    finally:
        db.close()
