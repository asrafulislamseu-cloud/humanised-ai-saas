import os
import json
import asyncio
import time
import random
import itertools
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Depends, HTTPException
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

# --- 1. ডাটাবেজ সেটআপ (Cloud PostgreSQL বা SQLite ফলব্যাক) ---
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_saas_platform.db")

# SQLite হলে check_same_thread আর্গুমেন্ট লাগবে, PostgreSQL হলে লাগবে না
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
    message_limit = Column(Integer, default=0)       # যেমন: 2500 বা 12000
    messages_used = Column(Integer, default=0)       # ব্যবহৃত মেসেজ কাউন্ট
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

app = FastAPI(title="Humanised AI SaaS Platform with Interview Mode & Smart Fallback", version="7.0")

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
    return {"message": "Humanised AI SaaS Platform is running smoothly with robust error handling!"}

# --- 3. স্মার্ট এআই কল উইথ ফুল ফলব্যাক, রিট্রাই ও সার্ভার ওভারলোড প্রোটেকশন ---
async def call_gemini_with_smart_fallback(user, contents, temp_val, max_tokens, is_stream=False):
    keys_to_try = [user.user_api_key]
    
    if user.is_premium and MASTER_API_KEYS:
        for _ in range(min(3, len(MASTER_API_KEYS))):
            keys_to_try.append(next(master_key_cycle))

    last_exception = None
    
    for index, api_key in enumerate(keys_to_try):
        for attempt in range(2): # প্রতি কি-র জন্য সর্বোচ্চ ২ বার রিট্রাই চেষ্টা
            try:
                async with ai_semaphore:
                    client = genai.Client(api_key=api_key)
                    if is_stream:
                        response_stream = client.models.generate_content_stream(
                            model="gemini-3.6-flash",
                            contents=contents,
                            config=types.GenerateContentConfig(temperature=temp_val, max_output_tokens=max_tokens)
                        )
                        return response_stream
                    else:
                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=contents,
                            config=types.GenerateContentConfig(temperature=temp_val, max_output_tokens=max_tokens)
                        )
                        return response.text.strip()
            except Exception as e:
                error_str = str(e)
                last_exception = e
                
                # যদি রেট লিমিট বা কোটা শেষ হয়, তবে সাথে সাথে ফলব্যাক কী-তে সুইচ করবে
                if index == 0 and any(err in error_str for err in ["429", "ResourceExhausted", "Quota", "503", "ServiceUnavailable"]):
                    break 
                
                # অন্যান্য টেম্পোরারি নেটওয়ার্ক এররের জন্য ২ সেকেন্ড অপেক্ষা করে একবার রিট্রাই করবে
                if attempt == 0:
                    await asyncio.sleep(2)
                    continue
                else:
                    break

    raise last_exception

# --- 4. ইউজার রেজিস্ট্রেশন বা লগইন রাউটার ---
class UserRegisterRequest(BaseModel):
    email: str
    user_api_key: str

@app.post("/register-or-login")
def register_or_login(data: UserRegisterRequest, db: Session = Depends(get_db)):
    if not data.user_api_key or not data.user_api_key.strip():
        raise HTTPException(status_code=400, detail="Please provide your Gemini API key.")
    
    user = db.query(UserDB).filter(UserDB.email == data.email).first()
    if not user:
        user = UserDB(email=data.email, user_api_key=data.user_api_key.strip(), is_premium=False, message_limit=0)
        db.add(user)
    else:
        user.user_api_key = data.user_api_key.strip()
    
    db.commit()
    return {"status": "success", "message": "Successfully logged in with API key!"}

# --- professional_bio অপশনাল করা হয়েছে ---
class UserProfileUpdate(BaseModel):
    email: str
    professional_bio: Optional[str] = None

@app.post("/update-profile")
def update_user_profile(data: UserProfileUpdate, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    if data.professional_bio is not None:
        user.professional_bio = data.professional_bio
        db.commit()
        
    return {"status": "success", "message": "Professional profile/bio updated successfully!"}

class SubscriptionActivateRequest(BaseModel):
    email: str
    package_type: str

@app.post("/activate-subscription")
def activate_subscription(data: SubscriptionActivateRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Please log in or create an account first.")
        
    if data.package_type == "1_dollar":
        user.message_limit = 2500
    elif data.package_type == "4_dollar":
        user.message_limit = 12000
    else:
        raise HTTPException(status_code=400, detail="Invalid package type!")
        
    user.messages_used = 0
    user.is_premium = True
    user.expiry_date = datetime.utcnow() + timedelta(days=30)
    
    db.commit()
    db.refresh(user)
    
    return {
        "status": "success",
        "message": "Subscription activated successfully for 30 days!",
        "message_limit": user.message_limit,
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

# --- 6. মূল এআই প্রসেসিং রাউটার (টোকেন লিমিট বৃদ্ধি করা হয়েছে যাতে উত্তর অর্ধেক কেটে না যায়) ---
@app.post("/process-ai")
async def process_ai_request(
    email: str = Form(...),
    mode: ModeEnum = Form(...),
    persona: Optional[PersonaEnum] = Form(PersonaEnum.Mother),
    target_language: Optional[LanguageEnum] = Form(LanguageEnum.Bengali),
    user_message: Optional[str] = Form(""),
    slide_content: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    try:
        user = db.query(UserDB).filter(UserDB.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
            
        if not user.is_premium or (user.expiry_date and datetime.utcnow() > user.expiry_date):
            user.is_premium = False
            db.commit()
            raise HTTPException(status_code=403, detail="Please buy subscription for more chat")
            
        if user.messages_used >= user.message_limit:
            raise HTTPException(status_code=403, detail="Your message quota has been exhausted. Please renew your subscription.")

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
                f"Instructions: Give a confident, natural, and professional answer reflecting the candidate's background and details where applicable. If the question requires broader technical knowledge beyond the candidate's bio, use your expert intelligence to answer accurately."
            )
            max_tokens = 1500  # উত্তর যেন কেটে না যায় সেজন্য টোকেন বাড়িয়ে দেওয়া হলো
            temp_val = 0.3
        else:
            assigned_voice_file = USER_VOICE_SETTINGS.get(persona_val, "default_voice")
            prompt = (
                f"Act warmly and naturally as: {persona_val}. Language: {lang_val}. "
                f"Message: {user_message}"
            )
            max_tokens = 1500  # উত্তর যেন কেটে না যায় সেজন্য টোকেন বাড়িয়ে দেওয়া হলো
            temp_val = 0.5

        contents.append(prompt)
        
        ai_response_text = await call_gemini_with_smart_fallback(user, contents, temp_val, max_tokens, is_stream=False)
        
        user.messages_used += 1
        db.commit()
        
        remaining = user.message_limit - user.messages_used

        return {
            "status": "success",
            "assigned_voice_file": assigned_voice_file,
            "response": ai_response_text,
            "remaining_messages": remaining
        }
            
    except Exception as e:
        return {"error": str(e)}

# --- 7. ওয়েবসকেট এন্ডপয়েন্ট (রিয়েল-টাইম স্ট্রিম ও নো-স্টোরেজ সরাসরি ফরওয়ার্ডিং) ---
active_tasks: Dict[WebSocket, asyncio.Task] = {}

async def handle_ai_stream(websocket: WebSocket, data: dict, db: Session):
    try:
        email = data.get("email")
        user = db.query(UserDB).filter(UserDB.email == email).first()
        
        # এখানে এক্সপায়ারি ডেট চেক সহ আপডেট করা হয়েছে
        if not user or not user.is_premium or (user.expiry_date and datetime.utcnow() > user.expiry_date) or user.messages_used >= user.message_limit:
            await websocket.send_json({"status": "error", "message": "Please buy subscription for more chat"})
            return

        mode = data.get("mode", "emotional_chat")
        persona = data.get("persona", "Mother")
        target_language = data.get("target_language", "bn")
        user_message = data.get("user_message", "")
        
        if mode == "presentation":
            user_bio = user.professional_bio if user.professional_bio else "No specific candidate bio provided."
            prompt = (
                f"You are attending a professional interview as the candidate. "
                f"Candidate's Profile: {user_bio}. "
                f"Language: {target_language}. "
                f"Question: {user_message}"
            )
        else:
            prompt = f"Act as {persona} in language {target_language}. Message: {user_message}"

        await websocket.send_json({"status": "started"})

        response_stream = await call_gemini_with_smart_fallback(user, [prompt], temp_val=0.5, max_tokens=1500, is_stream=True)
        
        for chunk in response_stream:
            if chunk.text:
                # লোকাল মেমোরিতে জমিয়ে না রেখে সাথে সাথে ইউজারের কাছে পাঠিয়ে দেওয়া হচ্ছে
                await websocket.send_json({"status": "streaming", "chunk": chunk.text})
            await asyncio.sleep(0.0001)
        
        user.messages_used += 1
        db.commit()

        await websocket.send_json({"status": "completed", "remaining_messages": user.message_limit - user.messages_used})
        
    except asyncio.CancelledError:
        await websocket.send_json({"status": "interrupted"})
        raise
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})

@app.websocket("/ws/live-ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
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

            task = asyncio.create_task(handle_ai_stream(websocket, data, db))
            active_tasks[websocket] = task
    except WebSocketDisconnect:
        if websocket in active_tasks and not active_tasks[websocket].done():
            active_tasks[websocket].cancel()
        print("WebSocket disconnected.")
    finally:
        db.close()