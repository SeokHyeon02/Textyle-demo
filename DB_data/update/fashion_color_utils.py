import os
import re
import sys
from io import BytesIO
from urllib.parse import urljoin

from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
VECTOR_SERVER_DIR = os.path.join(ROOT_DIR, "Textyle-vectorserver")
if VECTOR_SERVER_DIR not in sys.path:
    sys.path.insert(0, VECTOR_SERVER_DIR)

from fashion_color_extraction import (
    DENIM_COLOR_RGB_CENTROIDS,
    FASHION_TO_FINAL_COLOR,
    FINAL_COLOR_CATEGORIES,
    FINAL_COLOR_RGB_CENTROIDS,
)

COLOR_DB_COLUMN = "dominant_color"
COLOR_CONFIDENCE_DB_COLUMN = "color_confidence"
COLOR_CANDIDATES_DB_COLUMN = "color_candidates"
NAME_COLUMN = os.environ.get("IMAGE_VIEWER_NAME_COLUMN", "name")
IMAGE_COLUMN = os.environ.get("IMAGE_VIEWER_IMAGE_COLUMN", "image_url")
SUB_CATEGORY_COLUMN = os.environ.get("IMAGE_VIEWER_SUB_CATEGORY_COLUMN", "sub_category")

IMAGE_URL_PATTERN = re.compile(r"https?://[^\"'\s<>]+", re.IGNORECASE)
DIRECT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
MIN_COLOR_PIXEL_COUNT = 80
HIGH_COLOR_RATIO = 0.45
MEDIUM_COLOR_RATIO = 0.30
BACKGROUND_COLOR_DISTANCE = 48
MIXED_SECOND_COLOR_RATIO = 0.25

COLOR_RGB_CENTROIDS = FINAL_COLOR_RGB_CENTROIDS
DENIM_COLOR_CENTROIDS = DENIM_COLOR_RGB_CENTROIDS

COLOR_KEYWORDS = {
    "black": {"black", "blk", "bk", "블랙", "검정", "검정색", "검은색", "까만색", "흑청"},
    "white": {"white", "wht", "ivory", "화이트", "흰색", "하얀색", "아이보리"},
    "gray": {"gray", "grey", "charcoal", "그레이", "회색", "차콜"},
    "navy": {"navy", "nvy", "네이비", "남색"},
    "blue": {"blue", "sax", "sky blue", "블루", "파랑", "파란색", "중청", "연청"},
    "indigo": {
        "indigo",
        "raw denim",
        "dark denim",
        "dark blue",
        "인디고",
        "생지",
        "진청",
        "다크블루",
        "다크 블루",
    },
    "red": {"red", "burgundy", "wine", "레드", "빨강", "빨간색", "버건디"},
    "green": {"green", "mint", "olive", "올리브"},
    "khaki": {"khaki", "카키"},
    "yellow": {"yellow", "mustard", "옐로우", "노랑", "노란색"},
    "beige": {"beige", "cream", "sand", "oatmeal", "베이지", "크림", "샌드", "오트밀"},
    "brown": {"brown", "camel", "mocha", "브라운", "갈색", "카멜"},
    "pink": {"pink", "핑크", "분홍", "분홍색"},
    "purple": {"purple", "violet", "lavender", "퍼플", "보라", "보라색"},
    "orange": {"orange", "오렌지", "주황", "주황색"},
}

COLOR_PRIORITY = (
    "indigo",
    "khaki",
    "navy",
    "beige",
    "gray",
    "black",
    "white",
    "brown",
    "green",
    "blue",
    "red",
    "pink",
    "purple",
    "orange",
    "yellow",
)


def normalize_final_color(color: str):
    normalized = FASHION_TO_FINAL_COLOR.get(color, color)
    return normalized if normalized in FINAL_COLOR_CATEGORIES else None


def build_image_request_headers(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if "msscdn.net" in (url or ""):
        headers["Referer"] = "https://www.musinsa.com/"
    return headers


def is_direct_image_url(url: str):
    lower_url = (url or "").split("?")[0].lower()
    return lower_url.endswith(DIRECT_IMAGE_EXTENSIONS)


def extract_first_image_url_from_html(html: str, base_url: str):
    meta_patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    ]
    for pattern in meta_patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return urljoin(base_url, match.group(1))
    for match in IMAGE_URL_PATTERN.finditer(html):
        candidate = match.group(0)
        if is_direct_image_url(candidate):
            return candidate
    return None


def download_product_image(image_or_product_url: str):
    import requests

    if not image_or_product_url:
        raise ValueError("image_url is empty")

    first_response = requests.get(
        image_or_product_url,
        timeout=15,
        headers=build_image_request_headers(image_or_product_url),
    )
    first_response.raise_for_status()
    content_type = first_response.headers.get("content-type", "").lower()

    if "image/" in content_type or is_direct_image_url(image_or_product_url):
        return Image.open(BytesIO(first_response.content)).convert("RGB")

    image_url = extract_first_image_url_from_html(first_response.text, image_or_product_url)
    if not image_url:
        raise ValueError("Could not find an image URL from the product page")

    image_response = requests.get(image_url, timeout=15, headers=build_image_request_headers(image_url))
    image_response.raise_for_status()
    return Image.open(BytesIO(image_response.content)).convert("RGB")


def normalize_product_name_for_match(item_name: str):
    spaced_name = " ".join((item_name or "").lower().split())
    compact_name = re.sub(r"[^a-z0-9가-힣]+", "", spaced_name)
    return spaced_name, compact_name


def keyword_matches_product_name(keyword: str, spaced_name: str, compact_name: str):
    keyword_spaced = " ".join((keyword or "").lower().split())
    keyword_compact = re.sub(r"[^a-z0-9가-힣]+", "", keyword_spaced)
    if not keyword_spaced or not keyword_compact:
        return False

    if re.search(rf"(?<![a-z0-9가-힣]){re.escape(keyword_spaced)}(?![a-z0-9가-힣])", spaced_name):
        return True

    has_korean = bool(re.search(r"[가-힣]", keyword_compact))
    if has_korean and len(keyword_compact) >= 3 and keyword_compact in compact_name:
        return True
    if not has_korean and len(keyword_compact) >= 4 and keyword_compact in compact_name:
        return True
    return False


def extract_option_color_texts(item_name: str):
    name = item_name or ""
    option_texts = []
    for match in re.finditer(r"[\[\(（]([^\]\)）]+)[\]\)）]", name):
        option_texts.append(match.group(1))
    for delimiter in ("_", " - "):
        if delimiter in name:
            option_texts.append(name.rsplit(delimiter, 1)[-1])
    return [text.strip() for text in option_texts if text and text.strip()]


def find_color_matches_in_text(text: str):
    spaced_name, compact_name = normalize_product_name_for_match(text)
    if not compact_name:
        return []

    matched_colors = []
    for color in COLOR_PRIORITY:
        for keyword in COLOR_KEYWORDS.get(color, set()):
            if keyword_matches_product_name(keyword, spaced_name, compact_name):
                final_color = normalize_final_color(color)
                if final_color:
                    matched_colors.append(final_color)
                break
    return matched_colors


def classify_color_candidates_from_name(item_name: str, max_candidates: int = 3):
    option_matches = []
    for option_text in extract_option_color_texts(item_name):
        for color in find_color_matches_in_text(option_text):
            if color not in option_matches:
                option_matches.append(color)

    matched_colors = option_matches or find_color_matches_in_text(item_name)
    if not matched_colors:
        return []

    ordered_colors = matched_colors[:max_candidates]
    if len(ordered_colors) == 1:
        scores = [1.0]
    elif len(ordered_colors) == 2:
        scores = [0.65, 0.35]
    else:
        scores = [0.55, 0.30, 0.15]

    return [
        {
            "color": color,
            "score": scores[index],
            "source": "name",
            "confidence": "high" if index == 0 else "medium",
        }
        for index, color in enumerate(ordered_colors)
    ]


def classify_color_from_name(item_name: str):
    candidates = classify_color_candidates_from_name(item_name, max_candidates=1)
    return candidates[0]["color"] if candidates else None


def is_skin_like(r: int, g: int, b: int):
    brightness = (r + g + b) / 3
    return (
        brightness >= 90
        and r > 95
        and g > 40
        and b > 20
        and max(r, g, b) - min(r, g, b) > 15
        and abs(r - g) > 15
        and r > g
        and r > b
    )


def is_ignored_color_pixel(r: int, g: int, b: int):
    if r > 242 and g > 242 and b > 242:
        return True
    if r < 12 and g < 12 and b < 12:
        return True
    return is_skin_like(r, g, b)


def squared_rgb_distance(left, right):
    return sum((int(a) - int(b)) ** 2 for a, b in zip(left, right))


def iter_image_pixels(image: Image.Image):
    if hasattr(image, "get_flattened_data"):
        return image.get_flattened_data()
    return image.getdata()


def classify_color_by_rgb(rgb_color, centroids=None):
    r, g, b = [int(value) for value in rgb_color]
    brightness = (r + g + b) / 3
    spread = max(r, g, b) - min(r, g, b)
    blue_bias = b - max(r, g)
    warm_bias = r - min(g, b)

    if blue_bias >= 18 and b >= 120 and spread >= 35:
        return "blue"
    if b >= r + 12 and b >= g - 4 and brightness < 130:
        return "indigo"
    if 28 <= brightness <= 105 and warm_bias >= 5 and r >= g and r >= b:
        return "brown"
    if brightness >= 210 and spread < 28:
        return "white"

    centroids = centroids or COLOR_RGB_CENTROIDS
    return min(centroids, key=lambda color: squared_rgb_distance(rgb_color, centroids[color]))


def quantized_color_candidates(pixels):
    buckets = {}
    for r, g, b in pixels:
        key = (int(r) // 32, int(g) // 32, int(b) // 32)
        if key not in buckets:
            buckets[key] = [0, 0, 0, 0]
        buckets[key][0] += 1
        buckets[key][1] += int(r)
        buckets[key][2] += int(g)
        buckets[key][3] += int(b)

    total = len(pixels)
    candidates = []
    for count, red_sum, green_sum, blue_sum in buckets.values():
        candidates.append({
            "count": count,
            "ratio": count / total if total else 0.0,
            "rgb": (int(red_sum / count), int(green_sum / count), int(blue_sum / count)),
        })
    candidates.sort(key=lambda candidate: candidate["count"], reverse=True)
    return candidates


def collect_border_pixels(image: Image.Image, border_ratio: float = 0.08):
    resized = image.convert("RGB").resize((224, 224))
    width, height = resized.size
    border_x = max(1, int(width * border_ratio))
    border_y = max(1, int(height * border_ratio))
    pixels = []
    for y in range(height):
        for x in range(width):
            if border_x <= x < width - border_x and border_y <= y < height - border_y:
                continue
            pixels.append(resized.getpixel((x, y)))
    return pixels


def estimate_background_colors(image: Image.Image):
    border_pixels = collect_border_pixels(image)
    if len(border_pixels) < MIN_COLOR_PIXEL_COUNT:
        return []
    return [
        candidate["rgb"]
        for candidate in quantized_color_candidates(border_pixels)[:3]
        if candidate["ratio"] >= 0.12
    ]


def is_near_background_color(pixel, background_colors):
    if not background_colors:
        return False
    threshold = BACKGROUND_COLOR_DISTANCE ** 2
    return any(squared_rgb_distance(pixel, background_color) <= threshold for background_color in background_colors)


def is_denim_context(item_name: str = "", sub_category: str = "", material: str = None):
    text = f"{item_name or ''} {sub_category or ''} {material or ''}".lower()
    denim_terms = (
        "denim",
        "jean",
        "jeans",
        "raw denim",
        "dark denim",
        "데님",
        "청바지",
        "흑청",
        "진청",
        "중청",
        "연청",
    )
    return material == "denim" or any(term in text for term in denim_terms)


def extract_candidate_pixels(image: Image.Image):
    resized = image.convert("RGB").resize((224, 224))
    background_colors = estimate_background_colors(image)
    pixels = []
    fallback_pixels = []

    for r, g, b in iter_image_pixels(resized):
        if is_ignored_color_pixel(r, g, b):
            continue
        pixel = (r, g, b)
        fallback_pixels.append(pixel)
        if is_near_background_color(pixel, background_colors):
            continue
        pixels.append(pixel)

    return pixels if len(pixels) >= MIN_COLOR_PIXEL_COUNT else fallback_pixels


def color_candidates_from_pixels(pixels, centroids=None, max_candidates: int = 3):
    centroids = centroids or COLOR_RGB_CENTROIDS
    grouped = {}
    total = len(pixels)

    for candidate in quantized_color_candidates(pixels):
        color = classify_color_by_rgb(candidate["rgb"], centroids)
        grouped.setdefault(color, {"count": 0, "red_sum": 0, "green_sum": 0, "blue_sum": 0})
        count = candidate["count"]
        red, green, blue = candidate["rgb"]
        grouped[color]["count"] += count
        grouped[color]["red_sum"] += red * count
        grouped[color]["green_sum"] += green * count
        grouped[color]["blue_sum"] += blue * count

    ranked = sorted(grouped.items(), key=lambda row: row[1]["count"], reverse=True)
    candidates = []
    for color, values in ranked[:max_candidates]:
        ratio = values["count"] / total if total else 0.0
        if ratio < 0.08:
            continue
        candidates.append({
            "color": color,
            "score": ratio,
            "source": "image",
            "confidence": "high" if ratio >= HIGH_COLOR_RATIO else ("medium" if ratio >= MEDIUM_COLOR_RATIO else "low"),
            "rgb": (
                int(values["red_sum"] / values["count"]),
                int(values["green_sum"] / values["count"]),
                int(values["blue_sum"] / values["count"]),
            ),
        })
    return candidates


def dominant_color_group(pixels, centroids=None):
    candidates = color_candidates_from_pixels(pixels, centroids=centroids, max_candidates=1)
    if not candidates:
        return None, None, 0.0, 0.0
    top = candidates[0]
    return top["color"], tuple(top["rgb"]), top["score"], 0.0


def classify_denim_color_from_pixels(pixels):
    if not pixels:
        return None
    denim_color, _avg_rgb, _top_ratio, _second_ratio = dominant_color_group(pixels, DENIM_COLOR_CENTROIDS)
    return denim_color


def extract_dominant_color_result(image: Image.Image, denim_context: bool = False):
    pixels = extract_candidate_pixels(image)
    if len(pixels) < MIN_COLOR_PIXEL_COUNT:
        return {
            "color": None,
            "confidence": "low",
            "reason": "not_enough_valid_pixels",
            "dominant_ratio": 0.0,
            "second_ratio": 0.0,
            "candidates": [],
        }

    if denim_context:
        denim_color = classify_denim_color_from_pixels(pixels)
        if denim_color:
            return {
                "color": denim_color,
                "confidence": "high",
                "reason": "denim_context",
                "dominant_ratio": 1.0,
                "second_ratio": 0.0,
                "candidates": [{"color": denim_color, "score": 1.0, "source": "image", "confidence": "high"}],
            }

    color_candidates = color_candidates_from_pixels(pixels)
    if not color_candidates:
        return {
            "color": None,
            "confidence": "low",
            "reason": "no_color_candidates",
            "dominant_ratio": 0.0,
            "second_ratio": 0.0,
            "candidates": [],
        }

    top_candidate = color_candidates[0]
    top_ratio = top_candidate["score"]
    second_ratio = color_candidates[1]["score"] if len(color_candidates) > 1 else 0.0
    if top_ratio >= HIGH_COLOR_RATIO and top_ratio - second_ratio >= 0.18:
        confidence = "high"
    elif top_ratio >= MEDIUM_COLOR_RATIO:
        confidence = "medium"
    else:
        confidence = "low"
    if confidence == "high" and second_ratio >= MIXED_SECOND_COLOR_RATIO:
        confidence = "medium"
        color_candidates[0]["confidence"] = "medium"

    return {
        "color": top_candidate["color"],
        "confidence": confidence,
        "reason": "",
        "dominant_ratio": top_ratio,
        "second_ratio": second_ratio,
        "candidates": [
            {key: value for key, value in candidate.items() if key != "rgb"}
            for candidate in color_candidates
        ],
    }


def trusted_image_color_candidates(color_result):
    return [
        candidate
        for candidate in color_result.get("candidates", [])
        if candidate.get("confidence") in {"high", "medium"}
    ][:3]


def extract_color_attributes(image: Image.Image, item):
    item_name = item.get(NAME_COLUMN) or ""
    name_candidates = classify_color_candidates_from_name(item_name)
    color_result = extract_dominant_color_result(
        image,
        denim_context=is_denim_context(item_name, item.get(SUB_CATEGORY_COLUMN) or ""),
    )
    image_candidates = trusted_image_color_candidates(color_result)

    if name_candidates and image_candidates:
        return {
            COLOR_DB_COLUMN: name_candidates[0]["color"],
            COLOR_CONFIDENCE_DB_COLUMN: color_result.get("confidence") or "low",
            COLOR_CANDIDATES_DB_COLUMN: image_candidates,
            "color_reason": "product_name",
        }

    if name_candidates:
        return {
            COLOR_DB_COLUMN: name_candidates[0]["color"],
            COLOR_CONFIDENCE_DB_COLUMN: color_result.get("confidence") or "low",
            COLOR_CANDIDATES_DB_COLUMN: name_candidates,
            "color_reason": "product_name_fallback",
        }

    if color_result.get("color") and color_result.get("confidence") in {"high", "medium"}:
        color_candidates = image_candidates
        if not color_candidates:
            color_candidates = [{
                "color": color_result.get("color"),
                "score": color_result.get("dominant_ratio", 1.0),
                "source": "image",
                "confidence": color_result.get("confidence"),
            }]
        return {
            COLOR_DB_COLUMN: color_result.get("color"),
            COLOR_CONFIDENCE_DB_COLUMN: color_result.get("confidence"),
            COLOR_CANDIDATES_DB_COLUMN: color_candidates,
            "color_reason": color_result.get("reason") or "image_pixels",
        }

    return {
        COLOR_DB_COLUMN: None,
        COLOR_CONFIDENCE_DB_COLUMN: None,
        COLOR_CANDIDATES_DB_COLUMN: [],
        "color_reason": color_result.get("reason") or "not_found_in_product_name_or_image",
    }
