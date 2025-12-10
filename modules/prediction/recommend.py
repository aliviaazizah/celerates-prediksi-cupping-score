def quality_category(score: float) -> str:
    if score >= 88:
        return "Premium"
    elif score >= 85:
        return "Specialty"
    elif score >= 80:
        return "Standar"
    else:
        return "Blend"

def recommendation_from_category(cat: str) -> str:
    if cat == "Premium":
        return "Ideal untuk ekspor / pasar spesial internasional."
    if cat == "Specialty":
        return "Cocok untuk kopi spesial, roasting medium."
    if cat == "Standar":
        return "Cocok untuk campuran / roasting harian."
    return "Disarankan untuk campuran / blend komersial. Periksa processing / defects untuk perbaikan."
