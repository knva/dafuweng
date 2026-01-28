import argparse
from re import A
from PIL import Image
import os
import io

COORDS = (1150, 657, 1210, 687)  # (left, top, right, bottom)


def crop_region(image_path: str, out_path: str, coords=COORDS):
    with Image.open(image_path) as im:
        # Ensure image is in RGBA/RGB
        im = im.convert('RGB')
        cropped = im.crop(coords)
        cropped.save(out_path)
        return out_path


def try_rapidocr(img_path: str):
    try:
        # Import the package
        from rapidocr_onnxruntime import RapidOCR
    except Exception as e:
        print(f"Failed to import rapidocr_onnxruntime: {e}")
        return None

    try:
        reader = RapidOCR()
    except Exception as e:
        print(f"Failed to initialize RapidOCR: {e}")
        return None

    # Try common API patterns
    api_calls = [
        lambda r, p: r.ocr(p),
        lambda r, p: r.recognize(p),
        lambda r, p: r.readtext(p),
        lambda r, p: r(p),
    ]

    for call in api_calls:
        try:
            res = call(reader, img_path)
            if res:
                return res
        except Exception:
            continue

    print("No usable API method found on RapidOCR instance.")
    return None


def parse_result(res):
    # Try to extract text from common return formats
    texts = []
    if res is None:
        return ""

    # If it's a dict with 'text' or similar
    if isinstance(res, dict):
        if 'text' in res:
            return res['text']
        # flatten values
        for v in res.values():
            if isinstance(v, str):
                texts.append(v)

    # If it's a list of detections
    if isinstance(res, list):
        for item in res:
            # item might be (box, text, score) or [box, text, score]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                txt = item[1]
                if isinstance(txt, (list, tuple)) and len(txt) >= 1:
                    txt = txt[0]
                texts.append(str(txt))
            elif isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])

    # Fallback to string
    if not texts:
        return str(res)

    return ' | '.join(t for t in texts if t)


def ocr(image_bytes: bytes) -> bool:
    """
    检查截图区域是否存在"自动"两个字

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

    # 从字节数据中裁剪指定区域
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # 转换为RGB
            img = img.convert('RGB')
            # 裁剪指定区域
            cropped = img.crop(COORDS)

            # 保存裁剪后的图片到内存
            buffer = io.BytesIO()
            cropped.save(buffer, format='PNG')
            crop_bytes = buffer.getvalue()
    except Exception as e:
        print(f"Failed to crop image: {e}")
        return False

    # 使用OCR识别
    try:
        reader = RapidOCR()
    except Exception as e:
        print(f"Failed to initialize RapidOCR: {e}")
        return False

    # 识别裁剪后的区域
    api_calls = [
        lambda r, p: r.ocr(p),
        lambda r, p: r.recognize(p),
        lambda r, p: r.readtext(p),
        lambda r, p: r(p),
    ]

    for call in api_calls:
        try:
            res = call(reader, crop_bytes)
            if res:
                # 解析识别结果
                text = parse_result(res)
                print(f"OCR识别结果: {text}")

                # 检查是否包含"自动"
                if '自动' in text:
                    print("检测到'自动'二字，跳过AI分析")
                    return True
                else:
                    return False
        except Exception:
            continue

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to source image (screenshot)')
    parser.add_argument('--out', help='Cropped output path', default='crop_region.png')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    crop_path = crop_region(args.image, args.out)
    print(f"Cropped region saved to: {crop_path}")

    res = try_rapidocr(crop_path)
    parsed = parse_result(res)
    print("Recognized text:")
    print(parsed)


if __name__ == '__main__':
    main()
