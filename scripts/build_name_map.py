"""Build ChineseEnglish student name mapping from reference default_config.

Outputs data/student_name_map.json with format:
{
  "乃愛": "Noa",
  "乃愛(睡衣)": "Noa_(Pajama)",
  ...
}

Both Simplified and Traditional Chinese names are included.
The English names match the filename convention in data/captures/角色头像_crop/.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
reference = REPO / "study" / "ref" / "core" / "config" / "default_config.py"
CROP_DIR = REPO / "data" / "captures" / "角色头像_crop"
OUT = REPO / "data" / "student_name_map.json"

# Parse reference default_config — stored as STATIC_DEFAULT_CONFIG = '''{ ... }'''
text = reference.read_text("utf-8")
match = re.search(r"STATIC_DEFAULT_CONFIG\s*=\s*'''(.*?)'''", text, re.DOTALL)
if not match:
    raise RuntimeError("Could not parse STATIC_DEFAULT_CONFIG")
config = json.loads(match.group(1))
students = config["student_names"]
print(f"reference student entries: {len(students)}")

# List crop filenames to build allowed English names
crop_names = set()
if CROP_DIR.is_dir():
    for f in CROP_DIR.glob("*.png"):
        crop_names.add(f.stem)
print(f"Crop avatars available: {len(crop_names)}")

# Traditional Chinese ↔ Simplified Chinese conversion for common BA chars
# We'll use opencc if available, else manual table
try:
    import opencc
    s2t = opencc.OpenCC("s2t")
    t2s = opencc.OpenCC("t2s")
    HAS_OPENCC = True
    print("Using OpenCC for s2t/t2s conversion")
except ImportError:
    HAS_OPENCC = False
    print("OpenCC not available, using manual mapping")

# Suffix mapping: reference uses CN conventions, filenames use Global conventions
SUFFIX_MAP = {
    "泳装": "Swimsuit", "泳裝": "Swimsuit",
    "正月": "New_Year",
    "体操服": "Sportswear", "體操服": "Sportswear",
    "运动服": "Sportswear", "運動服": "Sportswear",
    "女仆": "Maid", "女僕": "Maid",
    "兔女郎": "Bunny_Girl",
    "圣诞": "Christmas", "聖誕": "Christmas",
    "礼服": "Dress", "禮服": "Dress",
    "啦啦队": "Cheerleader", "啦啦隊": "Cheerleader",
    "应援团": "Cheerleader", "應援團": "Cheerleader",
    "温泉": "Hot_Spring", "溫泉": "Hot_Spring",
    "露营": "Camping", "露營": "Camping",
    "偶像": "Idol",
    "便服": "Casual",
    "私服": "Casual",
    "旗袍": "Qipao",
    "睡衣": "Pajama",
    "导游": "Guide", "導遊": "Guide",
    "临战": "Battle", "臨戰": "Battle",
    "乐队": "Band", "樂隊": "Band",
    "制服": "School_Uniform",
    "骑行": "Riding", "騎行": "Riding",
    "魔法": "Magical",
    "打工": "Part-Timer",
}


def global_to_filename(global_name: str) -> str:
    """Convert 'Hina (Swimsuit)'  'Hina_(Swimsuit)'."""
    return global_name.replace(" (", "_(").replace(" ", "_")


def cn_to_possible_filenames(cn_name: str, global_name: str) -> list:
    """Generate possible crop filenames from CN + Global names."""
    candidates = []
    # Primary: from Global name
    fn = global_to_filename(global_name)
    candidates.append(fn)
    # Also try without special chars
    fn2 = fn.replace("＊", "")
    if fn2 != fn:
        candidates.append(fn2)
    # Try common variant suffixes: CampCamping, TrackSportswear, Cheer SquadCheerleader
    VARIANT_MAP = {
        "Camp": "Camping", "Track": "Sportswear",
        "Cheer_Squad": "Cheerleader",
    }
    for old, new in VARIANT_MAP.items():
        if old in fn:
            candidates.append(fn.replace(old, new))
    return candidates


# Build mapping: Chinese name  English filename
name_map = {}  # cn_name -> filename_stem
unmapped = []

for student in students:
    cn_name = student.get("CN_name", "")
    global_name = student.get("Global_name", "")
    if not cn_name or not global_name:
        continue

    # Normalize parentheses
    cn_name = cn_name.replace("（", "(").replace("）", ")")

    # Find matching crop filename
    candidates = cn_to_possible_filenames(cn_name, global_name)
    matched = None
    for cand in candidates:
        if cand in crop_names:
            matched = cand
            break

    if matched:
        # Map CN name (simplified)  filename
        name_map[cn_name] = matched

        # Also add Traditional Chinese variant if possible
        if HAS_OPENCC:
            tc = s2t.convert(cn_name)
            if tc != cn_name:
                name_map[tc] = matched
    else:
        unmapped.append((cn_name, global_name, candidates))

# Add special manual mappings for known Traditional Chinese names
# These are names where s2t conversion doesn't produce the exact game text
MANUAL_TC = {
    # Names where TC differs significantly from SC
    "亞子": "Ako",
    "亞子(禮服)": "Ako_(Dress)",
    "亞伽里": "Akari",
    "亞伽里(正月)": "Akari_(New_Year)",
    "乃愛": "Noa",
    "乃愛(睡衣)": "Noa_(Pajama)",
    "白子＊恐怖": "Shiroko＊Terror",
    # All favorites from app_config (ensure they map correctly)
    "若藻": "Wakamo",
    "若藻(泳裝)": "Wakamo_(Swimsuit)",
    "聖亞": "Seia",
    "聖亞(泳裝)": "Seia_(Swimsuit)",
    "櫻子": "Sakurako",
    "櫻子(偶像)": "Sakurako_(Idol)",
    "紗織": "Saori",
    "紗織(禮服)": "Saori_(Dress)",
    "理緒": "Rio",
    "渚": "Nagisa",
    "渚(泳裝)": "Nagisa_(Swimsuit)",
    "花子": "Hanako",
    "花子(泳裝)": "Hanako_(Swimsuit)",
    "光": "Hikari",
    "明日奈": "Asuna",
    "明日奈(制服)": "Asuna_(School_Uniform)",
    "愛麗絲": "Arisu",
    "愛麗絲(女僕)": "Arisu_(Maid)",
    "妃咲": "Kisaki",
    "茉莉": "Mari",
    "茉莉(偶像)": "Mari_(Idol)",
    "蓮見": "Hasumi",
    "蓮見(泳裝)": "Hasumi_(Swimsuit)",
    "佳奈": "Kanna",
    "佳奈(泳裝)": "Kanna_(Swimsuit)",
    "寧瑠": "Neru",
    "寧瑠(制服)": "Neru_(School_Uniform)",
    "望美": "Nozomi",
    "瀨奈": "Sena",
    "瀨奈(便服)": "Sena_(Casual)",
    "時": "Toki",
    "時(兔女郎)": "Toki_(Bunny_Girl)",
    "優香": "Yuuka",
    "優香(睡衣)": "Yuuka_(Pajama)",
    "優香(體操服)": "Yuuka_(Sportswear)",
    "佳代子": "Kayoko",
    "佳代子(正月)": "Kayoko_(New_Year)",
    # Common TC student names
    "一花": "Ichika",
    "一花(泳裝)": "Ichika_(Swimsuit)",
    "三千留": "Michiru",
    "三森": "Mimori",
    "三森(泳裝)": "Mimori_(Swimsuit)",
}
for tc, en in MANUAL_TC.items():
    if en in crop_names:
        name_map[tc] = en

print(f"\nMapped: {len(name_map)} namefilename entries")
print(f"Unmapped: {len(unmapped)} entries")
if unmapped[:5]:
    print("  Sample unmapped:")
    for cn, gl, cands in unmapped[:5]:
        print(f"    '{cn}'  '{gl}' (tried: {cands})")

# Save
OUT.write_text(json.dumps(name_map, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved to {OUT}")

# Quick verification against tick_0062 names
print("\n=== Verification against tick_0062 names ===")
test_names = ["乃愛", "乃愛(睡衣)", "亞伽里", "亞伽里(正月)", "亞子"]
for name in test_names:
    result = name_map.get(name, "NOT FOUND")
    print(f"  '{name}'  '{result}'")
