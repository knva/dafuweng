from playwright.sync_api import sync_playwright
import time
import threading
import msvcrt
import google.generativeai as genai
import base64
import re
import json
import argparse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google AI
import httpx
from ocr_region import ocr

# Configuration - 代理模型
API_KEY = os.getenv("API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")
TARGET_URL = os.getenv("TARGET_URL")

# 设置代理地址（根据你的实际代理修改）
PROXY_URL = os.getenv("PROXY_URL")
# 基准分辨率
BASE_WIDTH = int(os.getenv("BASE_WIDTH", 1280))
BASE_HEIGHT = int(os.getenv("BASE_HEIGHT", 720))

def convert_coordinates(base_x: int, base_y: int, current_width: int, current_height: int) -> tuple:
    """将1280x720基准分辨率的坐标转换为当前分辨率的坐标"""
    # 如果是基准分辨率，直接返回
    if current_width == BASE_WIDTH and current_height == BASE_HEIGHT:
        return base_x, base_y
    
    scale_x = current_width / BASE_WIDTH
    scale_y = current_height / BASE_HEIGHT
    
    new_x = int(base_x * scale_x)
    new_y = int(base_y * scale_y)
    
    print(f"坐标转换: ({base_x},{base_y}) -> ({new_x},{new_y}) [缩放: {scale_x:.2f}x{scale_y:.2f}]")
    return new_x, new_y

def draw_click_indicator(page, x, y, color):
    """在屏幕上绘制点击指示器（线程运行）"""
    try:
        page.evaluate(f"""
            (function() {{
                // Remove old indicator if exists
                const old = document.getElementById('click-indicator');
                if (old) old.remove();
                
                // Create new indicator
                const div = document.createElement('div');
                div.id = 'click-indicator';
                div.style.cssText = `
                    position: fixed;
                    left: {x - 15}px;
                    top: {y - 15}px;
                    width: 30px;
                    height: 30px;
                    border: 3px solid rgb({color});
                    border-radius: 50%;
                    background: rgba({color}, 0.3);
                    pointer-events: none;
                    z-index: 999999;
                    animation: pulse 0.5s ease-out;
                `;
                document.body.appendChild(div);
                
                // Add pulse animation
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes pulse {{
                        0% {{ transform: scale(1); opacity: 1; }}
                        100% {{ transform: scale(2); opacity: 0; }}
                    }}
                `;
                document.head.appendChild(style);
                
                // Remove after 2 seconds
                setTimeout(() => div.remove(), 2000);
            }})();
        """)
    except Exception as e:
        print(f"绘制指示器失败: {e}")



GAME_PROMPT_TEMPLATE = """你是一个游戏自动化助手，帮我玩大富翁游戏。

截图尺寸：{width} x {height} 像素
坐标系：左上角是(0,0)，x向右增加，y向下增加
重要：所有坐标均基于1280x720基准分辨率，系统会自动转换到当前分辨率

请仔细分析截图，找到需要点击的**按钮**位置。常见按钮包括：
- 骰子按钮（通常在屏幕右下方坐标1171，621）：如果是橙色骰子先长按切换成绿色，绿色骰子会自动走，当骰子按钮上显示免费时，长按它，变成绿色自动，在攻击城市的时候会有相似的按钮，但是不点，执行攻击城市命令
- 确定/确认/关闭按钮
- 猜拳选项（剪刀(555,650)/石头(635,650)/布(715,650)）
- 攻击城市轰炸目标（队列点击：308,227;427,127;319,463）
- 愿望选择(直接点击：455,450)
- 掠夺钱箱（队列点击：227,420;327,420;447,405;587,434;697,420）
- 蛋糕塔，点击左上角返回

严格按JSON格式回复，必须包含task字段标识任务类型：
- 单次点击：{{"task": "任务名", "x": 数字, "y": 数字}}
- 长按：{{"task": "骰子", "x": 数字, "y": 数字, "hold": true}}
- 多次顺序点击：{{"task": "任务名", "clicks": [...]}}
- 无需操作：{{"task": "等待", "action": "wait"}}

任务名示例：骰子、确认、猜拳、轰炸、愿望、掠夺、动画
"""



# 创建带代理的 httpx 客户端
http_client = httpx.Client(proxy=PROXY_URL)

genai.configure(
    api_key=API_KEY,
    transport='rest',
    client_options={'api_endpoint': API_ENDPOINT}
)

# 设置全局代理（通过环境变量方式）
# 设置全局代理（通过环境变量方式）
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL

model = genai.GenerativeModel(MODEL_NAME)

def decide_action_with_ai(image_bytes, viewport_width, viewport_height):
    """Sends the screenshot bytes (in-memory) to the AI and gets coordinates to click."""
    print(f"Sending screenshot bytes to AI...")

    # Format prompt with viewport size
    prompt = GAME_PROMPT_TEMPLATE.format(width=viewport_width, height=viewport_height)

    try:
        # Compress image to reduce size
        from PIL import Image
        import io

        with Image.open(io.BytesIO(image_bytes)) as img:
            # Scale down to 50%
            new_size = (img.width // 2, img.height // 2)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to JPEG with lower quality (keep colors)
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=50)
            image_data = buffer.getvalue()

        print(f"Compressed image size: {len(image_data) // 1024}KB")
        # 小于10kb直接等待
        if len(image_data) < 10240:
            return {"action": "wait", "task": "等待"}

        # Create image part for Gemini
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_data
        }

        response = model.generate_content([prompt, image_part])
        content = response.text
        print(f"AI Response: {content}")
        
        # Extract task name
        task_match = re.search(r'"task"\s*:\s*"([^"]+)"', content)
        task = task_match.group(1) if task_match else "unknown"
        
        # Check if AI says to wait
        if '"action"' in content and 'wait' in content.lower():
            print(f"AI says to wait... (task: {task})")
            return {"action": "wait", "task": task}
        
        # Check for multi-click array
        if '"clicks"' in content:
            import json
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*"clicks"\s*:\s*\[[^\]]+\][^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    clicks = data.get("clicks", [])
                    if clicks:
                        return {"clicks": [(c["x"], c["y"]) for c in clicks], "task": task}
                except:
                    pass
        
        # Parse single coordinate
        match = re.search(r'"x"\s*:\s*(\d+)', content)
        match_y = re.search(r'"y"\s*:\s*(\d+)', content)
        if match and match_y:
            x = int(match.group(1))
            y = int(match_y.group(1))
            # Check if hold/long press is needed
            hold = '"hold"' in content.lower() and 'true' in content.lower()
            return {"x": x, "y": y, "hold": hold, "task": task}
        
    except Exception as e:
        print(f"AI Request failed: {e}")
    
    return None
    
def decide_fixed_action(msginfo: str, viewport_width: int, viewport_height: int) -> dict:
    """Based on OCR text (msginfo), return a fixed action if applicable."""
    if not msginfo:
        return None
        
    import random
    
    # Keyword detection (case insensitive) -> Action
    msginfo = msginfo.strip()
        
    # 3. 愿望 (Wish) -> Single Click
    if "愿望" in msginfo:
        # 455,450
        print(f"Fixed Action: Detected '愿望', executing fixed click.")
        return {"x": 455, "y": 450, "task": "愿望-固定"}
 
    # 2. 攻击 (Attack) -> Fixed Sequence
    if "攻击" in msginfo:
        # 308,227; 427,127; 319,463
        print(f"Fixed Action: Detected '攻击', executing fixed sequence.")
        return {
            "task": "攻击-固定",
            "clicks": [
                (308, 227),
                (427, 127),
                (319, 463)
            ]
        }

        
    # 4. 掠夺 (Loot) -> Fixed Sequence
    if "掠夺" in msginfo:
        # 227,420; 327,420; 447,405; 587,434; 697,420
        print(f"Fixed Action: Detected '掠夺', executing fixed sequence.")
        return {
            "task": "掠夺-固定",
            "clicks": [
                 (227, 420),
                 (327, 420),
                 (447, 405),
                 (587, 434),
                 (697, 420)
            ]
        }
        
   # 1. 猜拳 (Rock-Paper-Scissors) -> Random
    if "猜拳" in msginfo or '擂台' in msginfo:
        # Options: 剪刀(555,650), 石头(635,650), 布(715,650)
        options = [
            {"name": "剪刀", "x": 555, "y": 650},
            {"name": "石头", "x": 635, "y": 650},
            {"name": "布", "x": 715, "y": 650}
        ]
        choice = random.choice(options)
        print(f"Fixed Action: Detected '猜拳', choosing Random -> {choice['name']}")
        return {"x": choice['x'], "y": choice['y'], "task": "猜拳-随机"}


    return None

class AutomationState:
    paused = False

def keyboard_listener():
    """监听键盘输入，控制暂停/恢复"""
    while True:
        if msvcrt.kbhit():
            # 读取按键，不回显
            try:
                key = msvcrt.getch()
                # 检查是否是 'p' 键 (支持大小写)
                if key.lower() == b'p':
                    AutomationState.paused = not AutomationState.paused
                    state = "暂停" if AutomationState.paused else "恢复"
                    print(f"\n[系统] 自动化已{state}。再次按下 'P' 键继续...")
            except Exception:
                pass
        time.sleep(0.1)


def main(browser_type="chromium"):
    print("Starting DA FU WENG (大富翁) automation...")

    with sync_playwright() as p:
        # Use persistent context to save cookies and session
        # 不同浏览器使用不同的数据目录
        if browser_type.lower() == "edge":
            user_data_dir = "./browser_data_edge"
            print("使用 Microsoft Edge 浏览器")
            context = p.chromium.launch_persistent_context(
                user_data_dir,
                headless=False,
                channel="msedge"
            )
        elif browser_type.lower() == "remote":
            print("连接远程浏览器 (ws://localhost:9222)")
            browser = p.chromium.connect_over_cdp("ws://localhost:9222")
            if browser.contexts:
                context = browser.contexts[0]
            else:
                context = browser.new_context()
        else:
            user_data_dir = "./browser_data"
            print("使用 Chromium 浏览器")
            context = p.chromium.launch_persistent_context(
                user_data_dir,
                headless=False
            )
        page = context.pages[0] if context.pages else context.new_page()
        
        try:
            print(f"Navigating to {TARGET_URL}...")
            page.goto(TARGET_URL)
            
            # Wait for manual login
            print("\n" + "="*50)
            print("请在浏览器中手动登录游戏")
            print("登录完成后，按回车键开始自动化...")
            print("="*50 + "\n")
            input()
            
            # 启动键盘监听线程
            threading.Thread(target=keyboard_listener, daemon=True).start()

            print("开始自动化循环，每秒截图一次。按 Ctrl+C 停止。")
            print("【提示】运行过程中按 'P' 键可以暂停/恢复自动化。")
            
            loop_count = 0
            last_task = None  # Track last task (no duplicate suppression)
            
            while True:
                if AutomationState.paused:
                    time.sleep(0.5)
                    continue

                loop_count += 1

                    # Capture screenshot to memory (bytes)
                image_bytes = page.screenshot()
                print(f"\n[Loop {loop_count}] Screenshot captured ({len(image_bytes) // 1024}KB)")
                
                # Get viewport size
                viewport = page.viewport_size
                vw, vh = viewport['width'], viewport['height']

                # 检查截图区域是否存在"自动"二字, 并获取OCR信息
                # ocr() returns (bool, str) -> (is_auto, msginfo)
                is_auto, msginfo = ocr(image_bytes)
                
                if is_auto:
                    print("检测到'自动'模式，跳过AI分析，等待下一轮...")
                    time.sleep(1)
                    continue

                # 优先检查是否存在固定逻辑 (Attack, Wish, Loot, RPS, etc.)
                fixed_coords = decide_fixed_action(msginfo, vw, vh)
                if fixed_coords:
                     print(f"Fixed action triggered: {fixed_coords.get('task')}")
                     coords = fixed_coords
                else:
                    # Get coordinates from AI (pass bytes, not a file)
                    coords = decide_action_with_ai(image_bytes, vw, vh)
                
                if coords:
                    current_task = coords.get("task", "unknown")
                    # No duplicate suppression: always execute the task even if same as last
                    last_task = current_task
                    print(f"Task: {current_task}")
                    
                    # Handle wait action
                    if coords.get("action") == "wait":
                        print("No action needed, waiting...")
                        time.sleep(1)
                        continue
                    image_bytes = page.screenshot()
                    ena,msginfo = ocr(image_bytes)
                    if ena:
                        print("检测到'自动'模式，跳过点击，等待下一轮...")
                        time.sleep(1)
                        continue
                    # Handle multi-click
                    if "clicks" in coords:
                        clicks = coords["clicks"]
                        print(f"Multi-click: {len(clicks)} positions")
                        for i, (cx, cy) in enumerate(clicks):
                            # 坐标转换
                            cx, cy = convert_coordinates(cx, cy, vw, vh)
                            print(f"  [{i+1}/{len(clicks)}] Clicking at ({cx}, {cy})")
                            
                            # Draw indicator
                            draw_click_indicator(page, cx, cy, "255, 0, 0")
                            
                            page.mouse.click(cx, cy)
                            time.sleep(1)  # 1 second interval between clicks
                        # Wait before next iteration
                        time.sleep(10)
                        continue
             
                    # Handle single click or hold
                    x = coords["x"]
                    y = coords["y"]
                    # 坐标转换
                    x, y = convert_coordinates(x, y, vw, vh)
                    hold = coords.get("hold", False)
                    
                    if hold:
                        print(f"Long pressing at ({x}, {y})")
                    else:
                        print(f"Clicking at ({x}, {y})")
                    
                    # Draw click indicator on page (green for hold, red for click)
                    indicator_color = "0, 255, 0" if hold else "255, 0, 0"
                    
                    # Use thread to draw indicator to avoid blocking
                    # Playwright sync API is not thread safe, calling directly (it's fast enough without sleep)
                    draw_click_indicator(page, x, y, indicator_color)
               
                    if hold:
                        # Long press: hold for 3 seconds
                        page.mouse.move(x, y)
                        page.mouse.down()
                        time.sleep(3)
                        page.mouse.up()
                    else:
                        page.mouse.click(x, y)
                else:
                    print("No action needed, waiting...")
            
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n自动化被用户停止。")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            context.close()
            print("浏览器已关闭。")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='大富翁游戏自动化')
    parser.add_argument('--browser', type=str, choices=['chromium', 'edge', 'remote'], default='chromium',
                        help='浏览器类型 (默认: chromium)')
    args = parser.parse_args()

    main(browser_type=args.browser)
