import argparse
from re import A
from PIL import Image, ImageOps, ImageEnhance
import os
import io
import time

COORDS = (1056, 524, 1250, 688)  # (left, top, right, bottom)


def get_ocr_reader():
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except Exception as e:
        print(f"Failed to initialize RapidOCR: {e}")
        return None


def smart_ocr(img: Image.Image, label: str = "Image") -> str:
    reader = get_ocr_reader()
    if not reader:
        return ""

    # Prepare image variants for OCR
    # 按照测试结果的成功率排序，优先使用效果最好的变体
    variants = []
    
    # 1. AutoContrast - 测试证明对 "自动" 识别效果最好
    try:
        auto_contrast = ImageOps.autocontrast(img, cutoff=5)
        variants.append(("AutoContrast", auto_contrast))
    except Exception:
        pass
    
    # 2. Equalized histogram - 测试证明也能识别 "自动"
    try:
        equalized = ImageOps.equalize(img)
        variants.append(("Equalized", equalized))
    except Exception:
        pass
    
    # 3. Upscaled + AutoContrast - 针对小文字
    try:
        upscaled = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
        upscaled_auto = ImageOps.autocontrast(upscaled, cutoff=5)
        variants.append(("Upscaled2xAuto", upscaled_auto))
    except Exception:
        pass
    
    # 4. Upscaled + Equalized
    try:
        upscaled = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
        upscaled_eq = ImageOps.equalize(upscaled)
        variants.append(("Upscaled2xEq", upscaled_eq))
    except Exception:
        pass
    
    # 5. Brightness boosted + Contrast
    try:
        bright_img = ImageEnhance.Brightness(img).enhance(1.5)
        bright_contrast = ImageEnhance.Contrast(bright_img).enhance(2.0)
        variants.append(("BrightContrast", bright_contrast))
    except Exception:
        pass
    
    # 6. Inverted + High Contrast - for semi-transparent dark overlays
    try:
        inverted = ImageOps.invert(img)
        inv_contrast = ImageEnhance.Contrast(inverted).enhance(3.0)
        variants.append(("InvContrast", inv_contrast))
    except Exception:
        pass
    
    # 7. Original
    variants.append(("Original", img))
    
    # 8. Upscaled Original (2x)
    try:
        upscaled = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
        variants.append(("Upscaled2x", upscaled))
    except Exception:
        pass
    
    # 9. High Contrast
    try:
        enhancer = ImageEnhance.Contrast(img)
        high_contrast = enhancer.enhance(2.5)
        variants.append(("HighContrast", high_contrast))
    except Exception:
        pass
    
    # 10. Inverted (Negative)
    try:
        inverted = ImageOps.invert(img)
        variants.append(("Inverted", inverted))
    except Exception:
        pass

    api_calls = [
        lambda r, p: r(p),  # Most common API
    ]

    for v_name, v_img in variants:
        # Convert variant to bytes
        v_buffer = io.BytesIO()
        # Ensure we save as PNG to preserve quality
        v_img.save(v_buffer, format='PNG')
        v_bytes = v_buffer.getvalue()
        
        for call in api_calls:
            try:
                res = call(reader, v_bytes)
                text = parse_result(res)
                if text.strip():
                    print(f"[{label}] ({v_name}) OCR识别结果: {text}")
                    return text
            except Exception:
                continue
    
    print(f"[{label}] 未能识别出任何文本")
    return ""


def crop_region(image_path: str, out_path: str, coords=COORDS):
    with Image.open(image_path) as im:
        # Ensure image is in RGBA/RGB
        im = im.convert('RGB')
        cropped = im.crop(coords)
        cropped.save(out_path)
        return out_path, cropped


def parse_result(res):
    """
    解析 RapidOCR 返回的结果。
    RapidOCR 通常返回 (result_list, elapsed_time) 元组。
    result_list 中每个元素是 [box, text, confidence]。
    """
    # 处理 RapidOCR 返回的 (result, elapsed_time) 元组
    if isinstance(res, tuple):
        res = res[0]  # 取第一个元素（识别结果）
    
    if res is None:
        return ""
    
    texts = []

    # 如果是检测结果列表
    if isinstance(res, list):
        for item in res:
            # item 格式: [[[x1,y1],[x2,y2],...], 'text', confidence]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # 第二个元素应该是文本
                txt = item[1]
                if isinstance(txt, str):
                    texts.append(txt)
            elif isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
    
    elif isinstance(res, dict):
        if 'text' in res:
            return res['text']
        for v in res.values():
            if isinstance(v, str):
                texts.append(v)
    
    elif isinstance(res, str):
        return res

    # 返回提取的文本，如果没有则返回空字符串
    if not texts:
        return ""

    return ' | '.join(t for t in texts if t)


def ocr(image_bytes: bytes) -> tuple[bool, str]:
    """
    检查截图区域是否存在"自动"两个字
    同时识别 400:0 到 900:400 范围内的文本并打印

    Args:
        image_bytes: 截图的字节数据

    Returns:
        bool: 如果检测到"自动"返回True，否则返回False
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as e:
        print(f"Failed to import rapidocr_onnxruntime: {e}")
        return False

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert('RGB')

            # 1. 识别消息区域 (400, 0, 900, 400)
            msginfo = ""
            try:
                msg_cropped = img.crop((400, 0, 900, 400))
                msginfo = smart_ocr(msg_cropped, "消息区域")
            except Exception as e:
                 print(f"Error processing message region: {e}")

            # 2. 识别自动按钮区域
            try:
                auto_cropped = img.crop(COORDS)
                text = smart_ocr(auto_cropped, "自动按钮")
            except Exception as e:
                print(f"Error processing auto button region: {e}")
                text = ""

            # Check for special events that require AI handling even if Auto is on
            special_keywords = ['愿望', '擂台', '攻击']
            has_special_event = any(k in msginfo for k in special_keywords)
            # print(f"[OCR] msginfo: {msginfo}, has_special_event: {has_special_event}")
            if text == "" and msginfo == "":
                print("无有效数据，跳过AI分析")
                return True, msginfo
            if '自' in text and "动"in text and not has_special_event and "长按" not in text and "以" not in text:
                print("检测到'自动'二字，跳过AI分析，等待下一轮...")
                return True, msginfo
            elif "掠夺了你的金库" in msginfo:
                print("检测到'掠夺了你的金库'，跳过AI分析，等待下一轮...")
                return True,""
            elif '自' in text and "动"in text and "拜访" in msginfo:
                print("检测到'拜访城市'，跳过AI分析，等待下一轮...")
                return True,""
            else:
                # timestamp = int(time.time())
                # save_dir = "screenshots"
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                # save_path = f"{save_dir}/{timestamp}-{msginfo[:2]}.png"
                
                # with open(save_path, "wb") as f:
                #     f.write(image_bytes)
                # print(f"Screenshot saved to {save_path}")
                return False, msginfo

    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return False, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to source image (screenshot)')
    parser.add_argument('--out', help='Cropped output path', default='crop_region.png')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    # Use default COORDS for main execution or maybe allow override?
    # For now, let's assume the user wants to test the COORDS region or just general OCR?
    # The original script cropped COORDS.
    
    _, cropped_img = crop_region(args.image, args.out) 
    print(f"Cropped region saved to: {args.out}")

    print("--- Recognizing Default Region (Auto Button) ---")
    text = smart_ocr(cropped_img, "Main-Crop")
    print("Recognized text:")
    print(text)
    
    # Also test the message region if running from main with a full screenshot?
    # The user passed a full screenshot 'screenshots\1769571223.png'.
    # Let's try to crop the message region too for debugging purposes.
    try:
        with Image.open(args.image) as full_img:
            full_img = full_img.convert('RGB')
            print("\n--- Recognizing Message Region (400:0 - 900:400) ---")
            msg_crop = full_img.crop((400, 0, 900, 400))
            smart_ocr(msg_crop, "Main-Message-Region")
    except Exception as e:
        print(f"Could not process full image for message region: {e}")


if __name__ == '__main__':
    main()
