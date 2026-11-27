"""Conservative local OCR rules that assist uncertain supervised predictions.

These are deployment rules, separate from learned model decisions and research metrics.
All evidence comes from automatic OCR, never file names or annotations.
"""
import re

VERSION = "ocr-content-rules-v2"
PRICE = r"[$€£]\s*\d[\d,.]*"


def match_content(text):
    def has(pattern):
        return re.search(pattern,text,re.I|re.M)

    def result(category,reason,pattern):
        quote = next((line for line in text.splitlines() if re.search(pattern,line,re.I)),"")
        return dict(label=category,rule=reason,evidence=quote[:240],version=VERSION)

    message = r"\b(?:RCS\s+(?:message|chat)|SMS\s*/?\s*MMS|iMessage|texting with|type a message|write a message|welcome to chat|join the chat)\b"
    if has(message):
        return result("chat","message_composer",message)
    if has(r"^\s*(?:messages|direct messages|chats)\s*$") and has(r"^\s*(?:you\s*:|add friends|unread|new message|search messages)"):
        return result("chat","conversation_inbox",r"messages|chats|you\s*:")

    confirmation = r"\b(?:payment (?:received|successful|completed)|order confirm(?:ed|ation)|booking confirm(?:ed|ation)|boarding pass)\b"
    if has(confirmation):
        return result("receipt_confirmation","confirmed_transaction",confirmation)
    if has(r"\b(?:receipt|invoice)\b") and has(PRICE+r"|\b(?:paid|total|invoice number)\b"):
        return result("receipt_confirmation","receipt_or_invoice",r"receipt|invoice|total")
    if has(r"\btickets?\b") and has(r"add to wallet|print (?:stage pass|selected tickets)|select ticket|(?:not )?valid for"):
        return result("receipt_confirmation","issued_ticket",r"ticket|add to wallet|stage pass")

    # A map can contain a small advertisement. Repeated street labels and navigation
    # controls take precedence over a lone offer embedded in the map.
    streets = re.findall(r"^.{0,45}\b(?:street|avenue|road|rd|ave|dr|boulevard)\s*$",text,re.I|re.M)
    navigation = r"\b(?:get directions|real time traffic|plan trip|arrive by|depart at)\b"
    if len(set(s.casefold() for s in streets))>=3 or has(navigation):
        return result("map_place","map_or_transit",navigation+r"|\b(?:street|avenue|road|rd|ave|dr)\b")
    lodging = r"\b(?:hotels|places to stay|guest favorite|destinations for you|check.in|check.out)\b"
    if len(re.findall(lodging,text,re.I))>=2:
        return result("map_place","lodging_recommendations",lodging)
    food = r"\b(?:curry|noodles?|rice|soup|tofu|vegetarian|entree|pasta|restaurant|diner|cuisine)\b"
    if len(set(w.casefold() for w in re.findall(food,text,re.I)))>=3 and has(PRICE):
        return result("map_place","restaurant_menu",food)

    purchase = r"\b(?:add to (?:cart|bag|basket)|buy now|shop now|proceed to checkout|black friday|flash sale)\b"
    if has(purchase):
        return result("shopping_product","purchase_or_sale",purchase)
    discount = r"\b\d{1,2}\s*%\s*off\b"
    retail = r"\b(?:sponsored|free shipp\w*|in stock|delivery|sale|discount|checkout|product|cart)\b"
    if has(PRICE) and (has(discount) or len(re.findall(PRICE,text))>=2 and has(retail)):
        return result("shopping_product","priced_product_offer",discount+r"|"+retail+r"|"+PRICE)
    if len(re.findall(PRICE,text))>=2 and len(re.findall(r"\b(?:traveler|seats|economy|business/first|options from|points)\b",text,re.I))>=3:
        return result("shopping_product","flight_fare_offer",r"options from|economy|"+PRICE)

    math = len(re.findall(r"[A-Za-z][A-Za-z0-9]*(?:\([^\n)]{1,20}\))?\s*=",text))
    if math>=3 and has(r"iteration|theorem|matrix|polynomial|equation|set of all|definition"):
        return result("article_reading","educational_material",r"iteration|theorem|matrix|polynomial|equation|set of all|definition")
    games = r"\b(?:battle day|attacks remaining|find a random opponent|teambuilder|game freak)\b"
    if has(games):
        return result("social_meme","game_screen",games)
    return None
