import pytest
from classifier import production
from classifier.content_rules import match_content


@pytest.mark.parametrize("text,expected",[
    ("RCS message\nThis sale is 50% off", "chat"),
    ("Messages\nAdd Friends\nYou: Hello there", "chat"),
    ("Welcome to chat!\nJoin the chat now\nHello everyone", "chat"),
    ("Cotton shirt\n$48\nAdd to bag", "shopping_product"),
    ("Sponsored\nFan\n$29.29\n$55.59\nFree shipping", "shopping_product"),
    ("Black Friday Deal\n35% off", "shopping_product"),
    ("Payment received\nTotal $18.50", "receipt_confirmation"),
    ("Add to Wallet\nPrint Stage Pass\nTickets", "receipt_confirmation"),
    ("Extra Ticket - Not Valid for\nIndoor Ceremony", "receipt_confirmation"),
    ("Hotels\nPlaces to stay\nGuest favorite\nBuy now\nMessages", "map_place"),
    ("Vegetarian curry\nTofu\nNoodle soup\n$17.00\n$15.00", "map_place"),
    ("Main Rd\nWillow Ave\nShelley Dr\nBlack Friday\n40% off", "map_place"),
    ("Battle Day\nAttacks remaining\n$300", "social_meme"),
    ("Receipt saved in Drive\nFolder\nSettings",None),
    ("Messages\nProfile\nTrips\nWelcome",None),
    ("Matrix iteration\nx0=3\nx1=8\nx2=15", "article_reading"),
])
def test_local_content_rules(text,expected):
    found = match_content(text)
    assert (found["label"] if found else None)==expected


@pytest.mark.parametrize("detector_label,detector_accepted",[("False",True),("True",False)])
def test_detector_does_not_block_content(monkeypatch,detector_label,detector_accepted):
    f={"sha256":"fixture","status":"ok","ocr":{"text":"Hello there\nSee you tomorrow"}}
    monkeypatch.setattr(production,"extract",lambda *args:f)
    def decision(head,features):
        if head=="detector":
            return dict(label=detector_label,accepted=detector_accepted)
        return dict(label="chat",accepted=True)
    monkeypatch.setattr(production,"decision",decision)
    r=production.predict_image({"tasks":{"is_screenshot":"detector","content_type":"content"}},b"image")
    assert r["route"]=="chat" and r["category_source"]=="supervised_model"


def test_rules_correct_sales_and_uncertain_content_is_miscellaneous(monkeypatch):
    f={"sha256":"fixture","status":"ok","ocr":{"text":"Fan\n$29\nAdd to cart"}}
    monkeypatch.setattr(production,"extract",lambda *args:f)
    monkeypatch.setattr(production,"decision",lambda *args:dict(label="article_reading",accepted=False))
    bundle={"tasks":{"content_type":"content"}}
    r=production.predict_image(bundle,b"image")
    assert r["route"]=="shopping_product" and r["category_source"]=="ocr_rule"
    assert r["tasks"]["content_type"]["label"]=="article_reading"
    assert r["actions"][0]["type"]=="consider_purchase"
    f["ocr"]["text"]="Unclear content"
    r=production.predict_image(bundle,b"image")
    assert r["route"]=="other" and r["actions"]==[]
    f["status"]="failed"
    f["clip"]={"error":"Encoder failed"}
    f["ocr"]["status"]="ok"
    assert production.predict_image(bundle,b"image")["route"]=="Processing error"


def test_complete_taxonomy_uses_learned_category_before_ocr_rules(monkeypatch):
    from types import SimpleNamespace
    from classifier.schema import NAMES
    features={"sha256":"fixture","status":"ok","ocr":{"text":"Add to cart\n$50"}}
    monkeypatch.setattr(production,"extract",lambda *args:features)
    model=SimpleNamespace(classes_=list(NAMES))
    bundle={"tasks":{"content_type":{"model":model}}}
    for label in NAMES:
        monkeypatch.setattr(production,"decision",lambda *args,label=label:dict(label=label,accepted=True))
        result=production.predict_image(bundle,b"image")
        assert result["route"]==label and result["category_source"]=="supervised_model"
