"""Generate synthetic phone screens. The separately sourced NASA portrait is a real photo."""
from pathlib import Path
from shutil import copyfile
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).parent / "demo"
ROOT.mkdir(exist_ok=True)
screens = [
 ("01-shirt", "FIELD SUPPLY", "Cotton camp shirt", "Linen blend  /  $48", "Add to bag", "#286957"),
 ("02-Café notes", "NEIGHBORHOOD GUIDE", "Café Juniper", "A quiet lunch spot on Elm Street", "Save this place", "#af6139"),
 ("03-gym", "NORTHSIDE FITNESS", "Join Northside Gym", "Monthly membership  /  $35", "Explore membership", "#62478a"),
 ("04-workshop", "COMMUNITY STUDIO", "Pottery workshop", "Event date: December 12, 2026", "Register for class", "#83593f"),
 ("05-recipe", "KITCHEN JOURNAL", "Lemon chickpea soup", "Recipe  /  Simmer for 20 minutes", "Read recipe", "#457364"),
 ("06-receipt", "PAYMENT DETAILS", "Café Juniper", "Payment received  /  Total $18.50", "View receipt", "#536a75"),
 ("07-meme", "FRIENDS FEED", "Me opening one tab", "My browser: 47 more friends!", "Like    Comment    Share", "#7d5383")]
for name, brand, title, line, button, color in screens:
    im = Image.new("RGB", (620, 960), "#faf9f4")
    d = ImageDraw.Draw(im)
    def label(x,y,s,size,fill):
        d.text((x,y),s,font=ImageFont.load_default(size=size),fill=fill)
    label(32,20,"9:41",24,"#252926")
    label(470,20,"5G   89%",21,"#252926")
    d.rectangle((0,82,620,152),fill=color)
    label(32,101,brand,29,"white")
    d.rounded_rectangle((28,184,592,490),24,fill=color)
    d.ellipse((210,218,415,434),fill="#e9d9b8")
    if name=="01-shirt":
        d.polygon([(247,265),(278,246),(314,260),(348,246),(382,269),(368,310),(349,303),(349,385),(270,385),(270,303),(250,313)],fill="#426d79")
    label(32,540,title,40,"#222b28")
    label(32,625,line,28,"#35413e")
    d.rounded_rectangle((32,736,588,817),20,fill=color)
    label(65,756,button,29,"white")
    d.rounded_rectangle((212,932,408,939),4,fill="#303631")
    im.save(ROOT/(name+".png"))
Image.new("RGB",(360,240),"#c3b5a8").save(ROOT/"08-crop.webp")
(ROOT/"copies").mkdir(exist_ok=True)
copyfile(ROOT/"01-shirt.png",ROOT/"copies"/"shirt copy.png")
(ROOT/"broken.jpg").write_bytes(b"incomplete fixture")
(ROOT/"notes.txt").write_text("Synthetic demo screens and a public NASA sample portrait.")
