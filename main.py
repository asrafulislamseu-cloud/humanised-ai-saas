import os
import json
import asyncio
import time
import random
import itertools
import base64
import hmac
import hashlib
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Depends, HTTPException, Response, Cookie, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
from enum import Enum
from dotenv import load_dotenv
from google import genai
from google.genai import types
import httpx
from contextlib import asynccontextmanager
import edge_tts  # Edge TTS ইম্পোর্ট

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
    package_type = Column(String, nullable=True)     
    message_limit = Column(Integer, default=0)       
    messages_used = Column(Integer, default=0)       
    free_messages_used = Column(Integer, default=0)  
    edge_tts_used = Column(Integer, default=0)       
    expiry_date = Column(DateTime, nullable=True)    
    professional_bio = Column(Text, nullable=True)   
    last_reset_date = Column(String, nullable=True)  

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

# --- দৈনিক লিমিট রিসেট ফাংশন ---
def check_and_reset_daily_limits(user: UserDB, db: Session):
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if user.last_reset_date != today_str:
        user.edge_tts_used = 0
        user.last_reset_date = today_str
        db.commit()

# --- একই ইউজারের ডাবল রিকোয়েস্ট আটকানোর জন্য একটিভ রিকোয়েস্ট ট্র্যাকার ---
active_processing_users = set()

# --- 2. জেমিনি মাল্টি-এপিআই কি ও মাস্টার কি পুল সেটআপ ---
raw_user_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
if raw_user_keys:
    API_KEYS = [k.strip() for k in raw_user_keys.split(",") if k.strip()]
else:
    API_KEYS = []

raw_master_keys = os.getenv("GEMINI_MASTER_KEYS", "")
MASTER_API_KEYS = [k.strip() for k in raw_master_keys.split(",") if k.strip()]
master_key_cycle = itertools.cycle(MASTER_API_KEYS) if MASTER_API_KEYS else None

user_cooldown_tracker: Dict[str, float] = {}
COOLDOWN_DURATION = 65.0  

# সার্ভারের অতিরিক্ত চাপ এড়ানোর জন্য কিউ/কনকারেন্সি লিমিট (Semaphore)
MAX_CONCURRENT_AI_CALLS = 20
ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_CALLS)

app = FastAPI(title="Humanised AI SaaS Platform with Multi-Account Support", version="14.8", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ৭টি ভাষার জন্য নির্দিষ্ট Edge TTS ভয়েস ম্যাপিং ডিকশনারি ---
EDGE_VOICE_MAPPING = {
    "bn": {
        "Father": "bn-BD-PradeepNeural", "Mentor": "bn-BD-PradeepNeural", "Professional": "bn-BD-PradeepNeural",
        "Husband": "bn-IN-BashkarNeural", "Boyfriend": "bn-IN-BashkarNeural", "Mother": "bn-BD-NabanitaNeural",
        "Wife": "bn-IN-TanishaaNeural", "Girlfriend": "bn-IN-TanishaaNeural", "Friend": "bn-BD-PradeepNeural"
    },
    "en": {
        "Father": "en-US-ChristopherNeural", "Mentor": "en-US-BrianNeural", "Professional": "en-US-AndrewNeural",
        "Husband": "en-US-GuyNeural", "Boyfriend": "en-US-RyanNeural", "Mother": "en-US-JennyNeural",
        "Wife": "en-US-AriaNeural", "Girlfriend": "en-US-AnaNeural", "Friend": "en-US-AndrewNeural"
    },
    "hi": {
        "Father": "hi-IN-MadhurNeural", "Mentor": "hi-IN-MadhurNeural", "Professional": "hi-IN-MadhurNeural",
        "Husband": "hi-IN-AaravNeural", "Boyfriend": "hi-IN-AaravNeural", "Mother": "hi-IN-SwaraNeural",
        "Wife": "hi-IN-AnanyaNeural", "Girlfriend": "hi-IN-AnanyaNeural", "Friend": "hi-IN-MadhurNeural"
    },
    "zh": {
        "Father": "zh-CN-YunxiNeural", "Mentor": "zh-CN-YunjianNeural", "Professional": "zh-CN-YunyangNeural",
        "Husband": "zh-CN-YunfengNeural", "Boyfriend": "zh-CN-YunxiaNeural", "Mother": "zh-CN-XiaoxiaoNeural",
        "Wife": "zh-CN-XiaoyiNeural", "Girlfriend": "zh-CN-XiaomoNeural", "Friend": "zh-CN-YunyangNeural"
    },
    "th": {
        "Father": "th-TH-NiwatNeural", "Mentor": "th-TH-NiwatNeural", "Professional": "th-TH-NiwatNeural",
        "Husband": "th-TH-AthitNeural", "Boyfriend": "th-TH-AthitNeural", "Mother": "th-TH-PremwadeeNeural",
        "Wife": "th-TH-AcharaNeural", "Girlfriend": "th-TH-AcharaNeural", "Friend": "th-TH-NiwatNeural"
    },
    "ar": {
        "Father": "ar-SA-HamedNeural", "Mentor": "ar-SA-HamedNeural", "Professional": "ar-SA-HamedNeural",
        "Husband": "ar-EG-ShakirNeural", "Boyfriend": "ar-EG-ShakirNeural", "Mother": "ar-SA-ZariyahNeural",
        "Wife": "ar-SA-MaryamNeural", "Girlfriend": "ar-EG-SalmaNeural", "Friend": "ar-SA-HamedNeural"
    },
    "es": {
        "Father": "es-ES-AlvaroNeural", "Mentor": "es-ES-AlvaroNeural", "Professional": "es-ES-AlvaroNeural",
        "Husband": "es-ES-DuarteNeural", "Boyfriend": "es-MX-DanteNeural", "Mother": "es-ES-ElviraNeural",
        "Wife": "es-ES-EstrellaNeural", "Girlfriend": "es-MX-DaliaNeural", "Friend": "es-ES-AlvaroNeural"
    }
}

class ModeEnum(str, Enum):
    emotional_chat = "emotional_chat"
    presentation = "presentation"

class PersonaEnum(str, Enum):
    Mother = "Mother"
    Father = "Father"
    Wife = "Wife"
    Girlfriend = "Girlfriend"
    Husband = "Husband"
    Boyfriend = "Boyfriend"
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
    return {"message": "AI Platform is running smoothly with Gemini 3.6 Flash Support!"}

async def generate_voice_from_edge(text_to_speak: str, lang_code: str, persona_val: str) -> bytes:
    lang_dict = EDGE_VOICE_MAPPING.get(lang_code, EDGE_VOICE_MAPPING["bn"])
    voice_name = lang_dict.get(persona_val, "bn-BD-PradeepNeural")
    
    communicate = edge_tts.Communicate(text_to_speak, voice_name)
    audio_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.extend(chunk["data"])
    return bytes(audio_bytes)

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
        raise HTTPException(status_code=429, detail="Hold up 65 second or buy a top up package")

    last_exception = None
    
    for key_type, api_key in keys_to_try:
        if key_type == "master" and (not user.is_premium or user.messages_used >= user.message_limit):
            continue

        try:
            async with ai_semaphore:
                client = genai.Client(api_key=api_key)
                if is_stream:
                    # Async Stream Generator Return
                    async def stream_generator():
                        response_stream = await client.aio.models.generate_content_stream(
                            model="gemini-3.6-flash",
                            contents=contents,
                            config=types.GenerateContentConfig(temperature=temp_val, max_output_tokens=max_tokens)
                        )
                        async for chunk in response_stream:
                            yield chunk

                    return stream_generator(), key_type
                else:
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(temperature=temp_val, max_output_tokens=max_tokens)
                    )
                    res_text = ""
                    if hasattr(response, "text") and response.text:
                        res_text = response.text
                    elif response.candidates:
                        res_text = response.candidates[0].content.parts[0].text
                    return res_text.strip(), key_type
        except Exception as e:
            error_str = str(e)
            last_exception = e
            if any(err in error_str for err in ["429", "503", "ResourceExhausted", "Quota", "ServiceUnavailable"]):
                if key_type == "user":
                    user_cooldown_tracker[user_email] = time.time()
            continue

    if not user.is_premium and is_user_in_cooldown:
        raise HTTPException(status_code=429, detail="Hold up 65 second or buy a top up package")
        
    raise HTTPException(
        status_code=503, 
        detail="Google AI is currently experiencing high demand (503 Service Unavailable). Please try again in a moment."
    )

class UserRegisterRequest(BaseModel):
    email: str
    password: Optional[str] = ""
    user_api_key: str

@app.post("/register-or-login")
def register_or_login(data: UserRegisterRequest, response: Response, db: Session = Depends(get_db)):
    if not data.user_api_key or not data.user_api_key.strip():
        raise HTTPException(status_code=400, detail="Please provide your Gemini API key.")
    
    user = db.query(UserDB).filter(UserDB.email == data.email).first()
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    if not user:
        user = UserDB(
            email=data.email, 
            password=data.password.strip() if data.password else "", 
            user_api_key=data.user_api_key.strip(), 
            is_premium=False, 
            message_limit=0,
            free_messages_used=0,
            edge_tts_used=0,
            last_reset_date=today_str
        )
        db.add(user)
    else:
        user.user_api_key = data.user_api_key.strip()
        if data.password:
            user.password = data.password.strip()
    
    db.commit()
    
    response.set_cookie(
        key="current_user_email",
        value=user.email,
        max_age=99 * 24 * 60 * 60,
        httponly=True,
        samesite="lax"
    )
    
    return {
        "status": "success", 
        "message": "Account added successfully!",
        "user": {
            "email": user.email,
            "user_api_key": user.user_api_key,
            "is_premium": user.is_premium,
            "package_type": user.package_type,
            "remaining_messages": max(0, user.message_limit - user.messages_used) if user.is_premium else 0
        }
    }

class ForgotPasswordRequest(BaseModel):
    email: str

@app.post("/forgot-password")
def forgot_password(data: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == data.email).first()
    if not user:
        return {"status": "success", "message": "If this email is registered, password reset instructions have been sent."}
    
    return {"status": "success", "message": "Password reset instructions sent to your email."}

@app.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="current_user_email")
    return {"status": "success", "message": "Logged out successfully!"}

@app.get("/check-auth")
def check_auth(current_user_email: Optional[str] = Cookie(None), db: Session = Depends(get_db)):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Not logged in.")
        
    user = db.query(UserDB).filter(UserDB.email == current_user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
        
    return {
        "status": "authenticated",
        "email": user.email,
        "is_premium": user.is_premium,
        "package_type": user.package_type,
        "remaining_messages": max(0, user.message_limit - user.messages_used) if user.is_premium else 0
    }

class UserProfileUpdate(BaseModel):
    professional_bio: Optional[str] = None

@app.post("/update-profile")
def update_user_profile(
    data: UserProfileUpdate, 
    current_user_email: Optional[str] = Cookie(None), 
    db: Session = Depends(get_db)
):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Not logged in.")
        
    user = db.query(UserDB).filter(UserDB.email == current_user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    if data.professional_bio is not None:
        user.professional_bio = data.professional_bio
        db.commit()
        
    return {"status": "success", "message": "Profile updated successfully!"}

class SubscriptionActivateRequest(BaseModel):
    package_type: str

@app.post("/activate-subscription")
def activate_subscription(
    data: SubscriptionActivateRequest, 
    current_user_email: Optional[str] = Cookie(None), 
    db: Session = Depends(get_db)
):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Not logged in.")
        
    user = db.query(UserDB).filter(UserDB.email == current_user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
        
    added_limit = 2500 if data.package_type == "1_dollar" else 12000 if data.package_type == "4_dollar" else 0
    if added_limit == 0:
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
        "message": "Subscription activated!",
        "remaining_messages": max(0, user.message_limit - user.messages_used)
    }

class CheckoutRequest(BaseModel):
    package_type: str
    user_email: Optional[str] = None

@app.post("/create-paddle-checkout")
async def create_paddle_checkout(data: CheckoutRequest, current_user_email: Optional[str] = Cookie(None)):
    try:
        user_email = data.user_email or current_user_email
        if not user_email:
            raise HTTPException(status_code=400, detail="User email is required.")

        price_id_1_dollar = os.getenv("PADDLE_PRICE_ID_1_DOLLAR", "pri_01h...") 
        price_id_4_dollar = os.getenv("PADDLE_PRICE_ID_4_DOLLAR", "pri_01h...")
        selected_price_id = price_id_1_dollar if data.package_type == "1_dollar" else price_id_4_dollar

        async with httpx.AsyncClient() as client:
            paddle_response = await client.post(
                "https://api.paddle.com/transactions",
                headers={
                    "Authorization": f"Bearer {os.getenv('PADDLE_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "items": [{"price_id": selected_price_id, "quantity": 1}],
                    "customer": {"email": user_email},
                    "custom_data": {"email": user_email, "package_type": data.package_type}
                }
            )
            if paddle_response.status_code in [200, 201]:
                checkout_url = paddle_response.json().get("data", {}).get("checkout", {}).get("url")
                return {"status": "success", "checkout_url": checkout_url}
            else:
                raise HTTPException(status_code=400, detail=paddle_response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/paddle-webhook")
async def paddle_webhook(request: Request, db: Session = Depends(get_db)):
    try:
        body_bytes = await request.body()
        paddle_signature = request.headers.get("Paddle-Signature", "")
        # আপনার Environment Variable-এর নামের সাথে মিলিয়ে PADDLE_WEBHOOK_SECRET দেওয়া হলো
        secret_key = os.getenv("PADDLE_WEBHOOK_SECRET", "")

        # Paddle HMAC Signature যাচাইকরণ (যদি Secret Key দেওয়া থাকে)
        if secret_key and paddle_signature:
            ts_str, h1_str = "", ""
            parts = paddle_signature.split(";")
            for part in parts:
                if part.startswith("ts="):
                    ts_str = part.split("=")[1]
                elif part.startswith("h1="):
                    h1_str = part.split("=")[1]

            if ts_str and h1_str:
                signed_payload = f"{ts_str}:{body_bytes.decode('utf-8')}"
                computed_hash = hmac.new(secret_key.encode('utf-8'), signed_payload.encode('utf-8'), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(computed_hash, h1_str):
                    raise HTTPException(status_code=401, detail="Invalid Paddle Webhook Signature.")

        event_json = json.loads(body_bytes.decode("utf-8"))
        event_type = event_json.get("event_type")
        data = event_json.get("data", {})
        
        if event_type in ["transaction.completed", "subscription.created", "subscription.activated"]:
            customer_email = data.get("customer", {}).get("email") or data.get("custom_data", {}).get("email")
            if customer_email:
                user = db.query(UserDB).filter(UserDB.email == customer_email).first()
                if user:
                    added_limit = 12000 if any("4" in str(item.get("product", {}).get("name", "")) for item in data.get("items", [])) else 2500
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
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_persona_behavior_rules(persona_val: str) -> str:
    if persona_val == "Father":
        return "Speak with a strict, protective, and authoritative tone like a real father."
    elif persona_val == "Mother":
        return "Speak with overflowing maternal warmth, deep affection, and soothing words like a real mother."
    elif persona_val in ["Wife", "Girlfriend", "Husband", "Boyfriend"]:
        return "Speak with natural romantic warmth, sweet jealousy, emotional attachment, and playful annoyance (abhiman)."
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
    active_email = current_user_email or cookie_user_email
    if not active_email:
        raise HTTPException(status_code=401, detail="No active account selected. Please log in.")

    if active_email in active_processing_users:
        raise HTTPException(status_code=429, detail="Previous message is still processing.")

    active_processing_users.add(active_email)
    try:
        user = db.query(UserDB).filter(UserDB.email == active_email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
            
        check_and_reset_daily_limits(user, db)

        if user.is_premium and user.expiry_date and datetime.utcnow() > user.expiry_date:
            user.is_premium = False
            user.package_type = None
            db.commit()

        mode_val = mode.value
        persona_val = persona.value if persona else "Mother"
        lang_val = target_language.value if target_language else "bn"

        contents = get_recent_chat_history(db, active_email)

        if file:
            file_bytes = await file.read()
            contents.append({"role": "user", "parts": [types.Part.from_bytes(data=file_bytes, mime_type=file.content_type)]})

        if mode_val == "presentation":
            user_bio = user.professional_bio if user.professional_bio else "No bio provided."
            prompt = f"Candidate Profile: {user_bio}. Language: {lang_val}. Topic: {slide_content}. Question: {user_message}. Give a direct, professional answer."
            max_tokens, temp_val = 800, 0.4  
        else:
            prompt = f"Act as: {persona_val}. Language: {lang_val}. Guidelines: {get_persona_behavior_rules(persona_val)} User Message: {user_message}."
            max_tokens, temp_val = 600, 0.5  

        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        ai_response_text, key_used = await call_gemini_with_smart_fallback(user, contents, temp_val, max_tokens, is_stream=False)
        
        db.add(ChatHistoryDB(user_email=active_email, role="user", message=user_message))
        db.add(ChatHistoryDB(user_email=active_email, role="model", message=ai_response_text))

        has_audio, encoded_audio_base64 = False, None
        if interaction_type == "Audio / Voice":
            try:
                audio_bytes = await generate_voice_from_edge(ai_response_text, lang_val, persona_val)
                if audio_bytes:
                    has_audio = True
                    encoded_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            except Exception:
                pass
            user.edge_tts_used += 1

        if key_used == "master":
            user.messages_used += 1
            
        db.commit()
        
        return {
            "status": "success",
            "active_user": active_email,
            "key_used": key_used,
            "response": ai_response_text,
            "has_audio": has_audio,
            "audio_base64": encoded_audio_base64,
            "remaining_messages": max(0, user.message_limit - user.messages_used) if user.is_premium else 0
        }
    finally:
        if active_email in active_processing_users:
            active_processing_users.remove(active_email)

# --- WebSocket লাইভ স্ট্রিম হ্যান্ডলার ---
active_tasks: Dict[WebSocket, asyncio.Task] = {}

async def handle_ai_stream(websocket: WebSocket, data: dict, current_user_email: Optional[str]):
    db = SessionLocal()  # প্রতিটি স্ট্রিম রিকোয়েস্টের জন্য ফ্রেশ সেশন
    try:
        if not current_user_email:
            await websocket.send_json({"status": "error", "message": "Not logged in."})
            return

        user = db.query(UserDB).filter(UserDB.email == current_user_email).first()
        if not user:
            await websocket.send_json({"status": "error", "message": "User not found."})
            return

        check_and_reset_daily_limits(user, db)
        persona = data.get("persona", "Mother")
        target_language = data.get("target_language", "bn")
        user_message = data.get("user_message", "")
        
        contents = get_recent_chat_history(db, current_user_email)
        prompt = f"Act as: {persona}. Language: {target_language}. User Message: {user_message}."
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        await websocket.send_json({"status": "started"})
        response_stream, key_used = await call_gemini_with_smart_fallback(user, contents, temp_val=0.5, max_tokens=800, is_stream=True)
        
        full_ai_response = ""
        # Async Iteration যাতে ইভেন্ট লুপ ব্লক না হয়
        async for chunk in response_stream:
            chunk_text = getattr(chunk, "text", "") or (chunk.candidates[0].content.parts[0].text if chunk.candidates else "")
            if chunk_text:
                full_ai_response += chunk_text
                await websocket.send_json({"status": "streaming", "chunk": chunk_text})
            await asyncio.sleep(0.0001)
        
        db.add(ChatHistoryDB(user_email=current_user_email, role="user", message=user_message))
        db.add(ChatHistoryDB(user_email=current_user_email, role="model", message=full_ai_response))
        if key_used == "master":
            user.messages_used += 1
        db.commit()

        await websocket.send_json({"status": "completed", "remaining_messages": max(0, user.message_limit - user.messages_used)})
    except Exception as e:
        db.rollback()
        await websocket.send_json({"status": "error", "message": str(e)})
    finally:
        db.close() # ডাটাবেজ সেশন ক্লিনআপ

@app.websocket("/ws/live-ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            active_email = data.get("current_user_email") or websocket.cookies.get("current_user_email")

            if websocket in active_tasks and not active_tasks[websocket].done():
                active_tasks[websocket].cancel()

            task = asyncio.create_task(handle_ai_stream(websocket, data, active_email))
            active_tasks[websocket] = task
    except WebSocketDisconnect:
        if websocket in active_tasks and not active_tasks[websocket].done():
            active_tasks[websocket].cancel()
