from fastapi import FastAPI
import asyncio
from google import genai
from google.genai import types

# --- রেন্ডার সার্ভারের জন্য অ্যাপ ইনিশিয়ালাইজেশন ---
app = FastAPI()

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
        # প্রিমিয়াম না হলে বা মাস্টার কোটা শেষ হলে মাস্টার কি স্কিপ করবে
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
                
                # যদি ইউজারের কি-তে রেট লিমিট বা কোটা এরর খায়:
                if key_type == "user":
                    if any(err in error_str for err in ["429", "ResourceExhausted", "Quota", "503", "ServiceUnavailable"]):
                        # ইউজার প্রিমিয়াম হলে এবং মাস্টার কোটা থাকলে লুপ ব্রেক করে মাস্টার কি-তে ফলব্যাক করবে
                        if user.is_premium and user.messages_used < user.message_limit:
                            break 
                        else:
                            # ফ্রি ইউজার হলে অথবা প্রিমিয়াম হয়েও মাস্টার কোটা শেষ থাকলে 
                            # ১ সেকেন্ড ওয়েট করে রি-ট্রাই করবে, আর লিমি트 শেষ হলে এক মিনিট পর এমনিতেই জেমিনি থেকে ঠিক হয়ে যাবে
                            if attempt == 0:
                                await asyncio.sleep(1)
                                continue
                            else:
                                break
                
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                else:
                    break

    raise last_exception
