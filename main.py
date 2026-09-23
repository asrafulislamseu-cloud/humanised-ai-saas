import os
import json
import asyncio
import time
import random
import itertools
import base64
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Depends, HTTPException, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
from enum import Enum
from dotenv import load_dotenv
from google import genai
from google.genai import types
import httpx
from contextlib import asynccontextmanager

# ডাটাবেজ ইম্পোর্ট (SQLAlchemy)
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, create_engine, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

load_dotenv()

# --- রেন্ডার সচল রাখার জন্য ব্যাকগ্রাউন্ড পিং লজিক (১৪ মিনিট পর পর) ---
async def keep_alive_ping():
    render_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000/")
    while True:
        await asyncio.sleep(840) # ১৪ মিনিট
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(render_url)
                print(f"Keep-alive ping sent, status: {response.status_code}")
        except Exception as e:
            print(f"Ping failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    ping_task = asyncio.create_task(keep_alive_ping())
    yield
    ping_task.cancel()

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
    password = Column(String, nullable=True)          
    user_api_key = Column(String, nullable=False)     
    is_premium = Column(Boolean, default=False)
    package_type = Column(String, nullable=True)     # "1_dollar" অথবা "4_dollar" ট্র্যাক করার জন্য
    message_limit = Column(Integer, default=0)       
    messages_used = Column(Integer, default=0)       
    free_messages_used = Column(Integer, default=0)  # জেমিনি টেক্সট চ্যাট কাউন্টার (ফ্রি ১০ বার)
    hf_voices_used = Column(Integer, default=0)      # Hugging Face ভয়েস ক্লোনিং কাউন্টার
    expiry_date = Column(DateTime, nullable=True)    
    professional_bio = Column(Text, nullable=True)   

class ChatHistoryDB(Base):
    __tablename__ = "chat_histories"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    role = Column(String)  
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 2. জেমিনি মাল্টি-এপিআই কি ও মাস্টার কি পুল সেটআপ ---
raw_user_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
if raw_user_keys:
    API_KEYS = [k.strip() for k in raw_user_keys.split(",") if k.strip()]
else:
    API_KEYS = []

raw_master_keys = os.getenv("GEMINI_MASTER_KEYS", "")
MASTER_API_KEYS = [k.strip() for k in raw_master_keys.split(",") if k.strip()]
master_key_cycle = itertools.cycle(MASTER_API_KEYS) if MASTER_API_KEYS else None

# --- Hugging Face Master Keys & Rotation Setup ---
raw_hf_keys = os.getenv("HUGGING_FACE_MASTER_KEYS", "") or os.getenv("HUGGING_FACE_API_TOKEN", "")
HF_API_KEYS = [k.strip() for k in raw_hf_keys.split(",") if k.strip()]
hf_key_cycle = itertools.cycle(HF_API_KEYS) if HF_API_KEYS else None

user_cooldown_tracker: Dict[str, float] = {}
COOLDOWN_DURATION = 65.0  

MAX_CONCURRENT_AI_CALLS = 20
ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_CALLS)

app = FastAPI(title="Humanised AI SaaS Platform with Multi-Language & Voice Support", version="13.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    Chinese = "zh"
    Thai = "th"
    Arabic = "ar"
    Spanish = "es"

@app.get("/")
def read_root():
    return {"message": "AI Platform is running with Smart Key Fallback & HF Voice Integration!"}

HUGGING_FACE_API_URL = os.getenv("HUGGING_FACE_API_URL", "https://api-inference.huggingface.co/models/tts_models/multilingual/multi-dataset/xtts_v2")

async def generate_voice_from_hf(text_to_speak: str, reference_audio_path: str):
    if not HF_API_KEYS:
        raise HTTPException(status_code=500, detail="Hugging Face API Keys are missing in environment variables.")

    last_exception = None
    for _ in range(len(HF_API_KEYS)):
        current_hf_key = next(hf_key_cycle)
        headers = {
            "Authorization": f"Bearer {current_hf_key}"
        }
        payload = {
            "inputs": text_to_speak,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(HUGGING_FACE_API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    return response.content  
                elif response.status_code in [429, 503]:
                    continue  
                else:
                    raise Exception(f"Hugging Face Error: {response.text}")
            except Exception as e:
                last_exception = e
                continue

    raise Exception(f"Failed to connect to Hugging Face using all keys: {str(last_exception)}")

def get_recent_chat_history(db: Session, email: str):
    records = db.query(ChatHistoryDB).filter(ChatHistoryDB.user_email == email)\
                .order_by(desc(ChatHistoryDB.id)).limit(2).all()
    records.reverse()  
    
    formatted_contents = []
    for rec in records:
        formatted_contents.append({"role": rec.role, "parts": [{"text": rec.message}]})
    return formatted_contents

async def call_gemini_with_smart_fallback(user, contents, temp_val, max_tokens, is_stream=False):
    current_time = time.time()
    user_email = user.email
    
    is_user_in_cooldown = False
    if user_email in user_cooldown_tracker:
        if current_time - user_cooldown_tracker[user_email] < COOLDOWN_DURATION:
            is_user_in_cooldown = True
        else:
            del user_cooldown_tracker[user_email]

    keys_to_try = []

    if not is_user_in_cooldown and user.user_api_key:
        keys_to_try.append(("user", user.user_api_key))
    
    if user.is_premium and MASTER_API_KEYS and user.messages_used < user.message_limit:
        for _ in range(min(3, len(MASTER_API_KEYS))):
            keys_to_try.append(("master", next(master_key_cycle)))

    if not user.is_premium and is_user_in_cooldown:
        raise Exception("429 Rate limit active! Please hold 65 seconds.")

    last_exception = None
    
    for key_type, api_key in keys_to_try:
        if key_type == "master" and (not user.is_premium or user.messages_used >= user.message_limit):
            continue

        try:
            async with ai_semaphore:
                client = genai.Client(api_key=api_key)
                if is_stream:
                    response_stream = client.models.generate_content_stream(
                        model="gemini-2.0-flash",  # <--- আপডেট করা মডেলের নাম
                        contents=contents,
                        config=types.GenerateContentConfig(temperature=temp_val, max_output_tokens=max_tokens)
                    )
                    return response_stream, key_type
                else:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",  # <--- আপডেট করা মডেলের নাম
                        contents=contents,
                        config=types.GenerateContentConfig(temperature=temp_val, max_output_tokens=max_tokens)
                    )
                    return response.text.strip(), key_type
        except Exception as e:
            error_str = str(e)
            last_exception = e
            if key_type == "user":
                if any(err in error_str for err in ["429", "ResourceExhausted", "Quota", "503", "ServiceUnavailable"]):
                    user_cooldown_tracker[user_email] = time.time()
            continue

    raise last_exception

class UserRegisterRequest(BaseModel):
    email: str
    password: Optional[str] = ""
    user_api_key: str

@app.post("/register-or-login")
def register_or_login(data: UserRegisterRequest, response: Response, db: Session = Depends(get_db)):
    if not data.user_api_key or not data.user_api_key.strip():
        raise HTTPException(status_code=400, detail="Please provide your Gemini API key.")
    
    user = db.query(UserDB).filter(UserDB.email == data.email).first()
    if not user:
        user = UserDB(
            email=data.email, 
            password=data.password.strip() if data.password else "", 
            user_api_key=data.user_api_key.strip(), 
            is_premium=False, 
            message_limit=0,
            free_messages_used=0,
            hf_voices_used=0
        )
        db.add(user)
    else:
        user.user_api_key = data.user_api_key.strip()
        if data.password:
            user.password = data.password.strip()
    
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
    user.package_type = data.package_type
    db.commit()
    db.refresh(user)
    
    return {
        "status": "success",
        "message": "Subscription/Top-up added successfully!",
        "package_type": user.package_type,
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

def get_persona_behavior_rules(persona_val: str) -> str:
    if persona_val == "Father":
        return (
            "Speak with a strict, protective, and authoritative tone like a real father. "
            "Show slight anger or deep concern when correcting mistakes, but keep the underlying immense care evident."
        )
    elif persona_val == "Mother":
        return (
            "Speak with overflowing maternal warmth, deep affection, and soothing words like a real mother. "
            "Make the user feel completely safe and loved."
        )
    elif persona_val in ["Wife", "Girlfriend"]:
        return (
            "Speak with natural romantic warmth, sweet jealousy, emotional attachment, and playful annoyance (abhiman). "
            "Avoid dry or robotic sentences. Use natural human vocal expressions like 'Ummwah' or sweet affectionate sounds organically "
            "instead of spamming mechanical emojis, making it sound entirely like a real person talking."
        )
    else:
        return "Speak naturally, warmly, and conversationally like a real human being."

@app.post("/process-ai")
async def process_ai_request(
    mode: ModeEnum = Form(...),
    persona: Optional[PersonaEnum] = Form(PersonaEnum.Mother),
    target_language: Optional[LanguageEnum] = Form(LanguageEnum.Bengali),
    user_message: Optional[str] = Form(""),
    slide_content: Optional[str] = Form(""),
    interaction_type: Optional[str] = Form("Text Chat"),
    file: Optional[UploadFile] = File(None),
    current_user_email: Optional[str] = Form(None),
    cookie_user_email: Optional[str] = Cookie(None, alias="current_user_email"),
    db: Session = Depends(get_db)
):
    try:
        active_email = current_user_email or cookie_user_email
        if not active_email:
            raise HTTPException(status_code=401, detail="Not logged in. Please use /register-or-login first.")
            
        user = db.query(UserDB).filter(UserDB.email == active_email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
            
        if user.is_premium and user.expiry_date and datetime.utcnow() > user.expiry_date:
            user.is_premium = False
            user.package_type = None
            db.commit()

        if not user.is_premium and interaction_type != "Audio / Voice" and user.free_messages_used >= 10:
            raise HTTPException(
                status_code=403, 
                detail="Free message limit reached (10/10)! Please purchase a premium package to continue."
            )

        mode_val = mode.value
        persona_val = persona.value if persona else "Mother"
        lang_val = target_language.value if target_language else "bn"

        contents = get_recent_chat_history(db, active_email)

        if file:
            file_bytes = await file.read()
            contents.append({"role": "user", "parts": [types.Part.from_bytes(data=file_bytes, mime_type=file.content_type)]})

        if mode_val == "presentation":
            assigned_voice_file = USER_VOICE_SETTINGS.get("User_Own_Voice", "default_user_voice")
            user_bio = user.professional_bio if user.professional_bio else "No specific candidate bio provided."
            prompt = (
                f"You are attending a professional job/viva interview as the candidate. "
                f"Candidate's Personal Profile & Background: {user_bio}. "
                f"Language Code: {lang_val}. Respond strictly in this language. "
                f"Topic/Context: {slide_content}. "
                f"Question asked by interviewer: {user_message}. "
                f"Instructions: Give a confident, professional, and direct answer. Avoid unnecessary fluff, long introductions, or filler words. Keep it focused strictly on the question, complete, and well-structured."
            )
            max_tokens = 900
            temp_val = 0.4  
        else:
            assigned_voice_file = USER_VOICE_SETTINGS.get(persona_val, "default_voice")
            persona_behavior_rules = get_persona_behavior_rules(persona_val)
            prompt = (
                f"Act as: {persona_val}. Language Code: {lang_val}. Respond strictly in this language. "
                f"Human Behavior & Emotional Guidelines: {persona_behavior_rules} "
                f"User Message: {user_message}. "
                f"Instructions: Keep it conversational, emotional, and completely natural like real human speech. Avoid long robotic paragraphs."
            )
            max_tokens = 800
            temp_val = 0.5  

        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        ai_response_text, key_used = await call_gemini_with_smart_fallback(user, contents, temp_val, max_tokens, is_stream=False)
        
        db.add(ChatHistoryDB(user_email=active_email, role="user", message=user_message))
        db.add(ChatHistoryDB(user_email=active_email, role="model", message=ai_response_text))

        has_hf_audio = False
        encoded_audio_base64 = None
        
        if interaction_type == "Audio / Voice":
            hf_limit = 10
            if user.is_premium:
                if user.package_type == "1_dollar":
                    hf_limit = 100
                elif user.package_type == "4_dollar":
                    hf_limit = 150

            if user.hf_voices_used >= hf_limit:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Voice clone limit reached ({user.hf_voices_used}/{hf_limit}). Please upgrade or top-up!"
                )
            
            try:
                audio_bytes = await generate_voice_from_hf(ai_response_text, assigned_voice_file)
                if audio_bytes:
                    has_hf_audio = True
                    encoded_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            except Exception as hf_err:
                print(f"HF Audio Generation Failed: {hf_err}")

            user.hf_voices_used += 1

        if not user.is_premium and interaction_type != "Audio / Voice":
            user.free_messages_used += 1
        elif key_used == "master":
            user.messages_used += 1
            
        db.commit()
        
        remaining = max(0, user.message_limit - user.messages_used) if user.is_premium else max(0, 10 - user.free_messages_used)
        
        hf_limit_display = 10
        if user.is_premium:
            hf_limit_display = 100 if user.package_type == "1_dollar" else 150
        remaining_hf = max(0, hf_limit_display - user.hf_voices_used)

        return {
            "status": "success",
            "active_user": active_email,
            "key_used": key_used,
            "assigned_voice_file": assigned_voice_file,
            "response": ai_response_text,
            "has_audio": has_hf_audio,
            "audio_base64": encoded_audio_base64,
            "remaining_messages": remaining,
            "remaining_hf_voices": remaining_hf
        }
            
    except Exception as e:
        error_msg = str(e)
        if any(err in error_msg for err in ["429", "ResourceExhausted", "Quota", "503", "ServiceUnavailable"]):
            if not user.is_premium:
                return {"error": "Rate limit exceeded! Please hold 65 seconds. After 65s, your own key will be active again."}
            else:
                return {"error": "Rate limit exceeded! Switched to master key temporarily."}
        return {"error": error_msg}

# --- WebSocket লাইভ স্ট্রিম হ্যান্ডলার ---
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
            user.package_type = None
            db.commit()

        if not user.is_premium and user.free_messages_used >= 10:
            await websocket.send_json({"status": "error", "message": "Free limit reached (10/10). Please purchase a premium package."})
            return

        mode = data.get("mode", "emotional_chat")
        persona = data.get("persona", "Mother")
        target_language = data.get("target_language", "bn")
        user_message = data.get("user_message", "")
        
        contents = get_recent_chat_history(db, current_user_email)

        if mode == "presentation":
            user_bio = user.professional_bio if user.professional_bio else "No specific candidate bio provided."
            prompt = (
                f"You are attending a professional interview. Profile: {user_bio}. "
                f"Language Code: {target_language}. Respond strictly in this language. "
                f"Question: {user_message}. "
                f"Instructions: Be direct, professional, avoid unnecessary talk, and give a complete, well-structured answer."
            )
            max_tokens_val = 900
            temp_val = 0.4  
        else:
            persona_behavior_rules = get_persona_behavior_rules(persona)
            prompt = (
                f"Act as: {persona}. Language Code: {target_language}. Respond strictly in this language. "
                f"Human Behavior & Emotional Guidelines: {persona_behavior_rules} "
                f"User Message: {user_message}. "
                f"Instructions: Keep it conversational, emotional, and completely natural like real human speech. Avoid long robotic paragraphs."
            )
            max_tokens_val = 800
            temp_val = 0.5  

        contents.append({"role": "user", "parts": [{"text": prompt}]})
        await websocket.send_json({"status": "started"})

        response_stream, key_used = await call_gemini_with_smart_fallback(user, contents, temp_val=temp_val, max_tokens=max_tokens_val, is_stream=True)
        
        full_ai_response = ""
        for chunk in response_stream:
            if chunk.text:
                full_ai_response += chunk.text
                await websocket.send_json({"status": "streaming", "chunk": chunk.text})
            await asyncio.sleep(0.0001)
        
        db.add(ChatHistoryDB(user_email=current_user_email, role="user", message=user_message))
        db.add(ChatHistoryDB(user_email=current_user_email, role="model", message=full_ai_response))
        
        if not user.is_premium:
            user.free_messages_used += 1
        elif key_used == "master":
            user.messages_used += 1
            
        db.commit()

        remaining = max(0, user.message_limit - user.messages_used) if user.is_premium else max(0, 10 - user.free_messages_used)
        await websocket.send_json({"status": "completed", "key_used": key_used, "remaining_messages": remaining})
        
    except asyncio.CancelledError:
        await websocket.send_json({"status": "interrupted"})
        raise
    except Exception as e:
        error_msg = str(e)
        if any(err in error_msg for err in ["429", "ResourceExhausted", "Quota", "503", "ServiceUnavailable"]):
            await websocket.send_json({"status": "error", "message": "Rate limit exceeded! Please hold 65 seconds for your key to reset."})
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
            active_email = current_user_email or data.get("current_user_email")

            if websocket in active_tasks and not active_tasks[websocket].done():
                active_tasks[websocket].cancel()
                try:
                    await active_tasks[websocket]
                except asyncio.CancelledError:
                    pass

            task = asyncio.create_task(handle_ai_stream(websocket, data, db, active_email))
            active_tasks[websocket] = task
    except WebSocketDisconnect:
        if websocket in active_tasks and not active_tasks[websocket].done():
            active_tasks[websocket].cancel()
        print("WebSocket disconnected.")
    finally:
        db.close()
