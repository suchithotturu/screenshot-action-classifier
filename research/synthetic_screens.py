"""Deterministic, fictional UI screenshots for a separate synthetic feature study.

No image model, Qwen, personal data or downloaded media is used. Labels are the
renderer intent, not independent human judgments. All classes share UI skins.
"""
import argparse
import hashlib
import json
import random
import textwrap
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from classifier.features import ROOT, atomic
from classifier.schema import NAMES

VERSION='synthetic-ui-v1'
SEED=51327
PEOPLE=['Alex','Morgan','Casey','Jamie','Taylor','Sam','Jordan','Riley','Avery','Drew']
PRODUCTS=['Trail headphones','Canvas backpack','Desk lamp','Running shoes','Travel mug','Reading chair','Cotton jacket','Wireless keyboard','Camera pouch','Weekend bag']
PLACES=['Juniper Cafe','Willow Gardens','Riverside Museum','Maple Kitchen','Northside Library','Hilltop Park','Harbor Market','Pine Trail']
PARAGRAPHS=[
 'A practical guide starts with a small experiment. Compare the alternatives, record what changes, and repeat the steps that worked.',
 'Before your next trip, leave time for a walk through the neighborhood. Local parks and small cafes can be useful places to stop.',
 'Keeping notes makes it easier to compare ideas later. Use clear headings and write one observation for each example.',
 'The design uses simple shapes and generous space. Consistent alignment makes long pages easier to scan on a phone.',
 'A useful routine can be built from short sessions. Choose a realistic goal, review the result, and make one adjustment at a time.',
 'When comparing products, consider durability, comfort and the features you will use. A lower price is only one part of the decision.',
]
JOKES=['I opened one tab. My browser invited forty friends.','Me: five more minutes. Also me: one more episode.','My weekend plans: recharge everything, including myself.','When the group finally agrees on a place to eat.','A very serious review of my favorite imaginary sandwich.','The playlist understood the assignment today.']


class Screen:
 def __init__(self,rng,family,index):
  self.r=rng; self.family=family
  self.w=[400,440,480,420,460][family%5]
  self.scale=self.w/480
  self.h=round(self.w*(2.0+0.04*(family%3)))
  dark=(index+family)%3==0
  self.bg='#141b26' if dark else '#f3f5f8'
  self.card='#253044' if dark else '#ffffff'
  self.fg='#eef2f6' if dark else '#172333'
  self.muted='#b4c0ce' if dark else '#66758a'
  self.accent=rng.choice(['#356bd8','#127d72','#8651ad','#aa5439','#476fa1','#965277'])
  self.im=Image.new('RGB',(self.w,self.h),self.bg)
  self.d=ImageDraw.Draw(self.im)
  self.pad=16+(family%4)*3
  self.radius=[8,14,20,24][family%4]
  self.text(20,9,f'{rng.randrange(1,13)}:{rng.randrange(0,60):02d}',14)
  self.text(370,9,'5G  86%',13)
 def xy(self,box): return tuple(round(x*self.scale) for x in box)
 def rect(self,box,fill=None,radius=None,outline=None):
  self.d.rounded_rectangle(self.xy(box),radius=round((self.radius if radius is None else radius)*self.scale),fill=fill or self.card,outline=outline)
 def text(self,x,y,s,size=18,fill=None,width=None):
  font=ImageFont.load_default(size=max(10,round(size*self.scale)))
  if width:
   words=s.split(); lines=[]; line=''
   for word in words:
    attempt=(line+' '+word).strip()
    if line and self.d.textlength(attempt,font=font)>width*self.scale:
     lines.append(line); line=word
    else: line=attempt
   if line:lines.append(line)
  else: lines=s.split('\n')
  for i,line in enumerate(lines): self.d.text(self.xy((x,y+i*(size+7))),line,font=font,fill=fill or self.fg)
  return y+len(lines)*(size+7)
 def header(self,title,subtitle=''):
  self.text(self.pad,48,'<' if self.family%2 else '=',25)
  self.text(57,46,title,24,width=340)
  if subtitle:self.text(57,78,subtitle,13,self.muted)
  self.text(438,48,':',24)
 def button(self,y,title):
  self.rect((self.pad,y,480-self.pad,y+49),self.accent,12)
  self.text(self.pad+16,y+12,title,19,'#ffffff')
 def footer(self,labels='Home     Explore     Saved     Profile'):
  y=self.h/self.scale-74
  self.rect((0,y,480,y+74),self.card,0)
  self.text(28,y+18,labels,14,self.muted)
  self.rect((176,y+58,304,y+63),self.muted,3)
 def art(self,box,kind='landscape'):
  x,y,x2,y2=box; self.rect(box,self.r.choice(['#c3d8d1','#e6d2bd','#c5d2e2','#d4c5dd']),12)
  if kind=='product':
   cx=(x+x2)/2;cy=(y+y2)/2; ww=(x2-x)*.33;hh=(y2-y)*.62
   self.rect((cx-ww/2,cy-hh/2,cx+ww/2,cy+hh/2),self.accent,16)
   self.d.arc(self.xy((cx-ww*.35,cy-hh*.7,cx+ww*.35,cy-hh*.2)),180,360,fill=self.fg,width=max(2,round(3*self.scale)))
   self.rect((cx-ww*.3,cy,cx+ww*.3,cy+hh*.25),'#dbe6ee',5)
  else:
   self.d.ellipse(self.xy((x+(x2-x)*.65,y+18,x+(x2-x)*.82,y+18+(x2-x)*.17)),fill='#f3d388')
   self.d.polygon([self.xy((x,y2)),self.xy((x+(x2-x)*.4,y+(y2-y)*.35)),self.xy((x2,y2))],fill='#658f8a')
   self.d.polygon([self.xy((x,y2)),self.xy((x+(x2-x)*.7,y+(y2-y)*.5)),self.xy((x2,y2))],fill='#3e6b75')


def render(label,family,index):
 seed=int(hashlib.sha256(f'{SEED}/{label}/{family}/{index}'.encode()).hexdigest()[:12],16)
 r=random.Random(seed);s=Screen(r,family,index);p=s.pad
 person=r.choice(PEOPLE);product=r.choice(PRODUCTS);place=r.choice(PLACES)
 price=r.randrange(12,180)+.99
 if label=='chat':
  inbox=family%4==1
  s.header('Inbox' if inbox else person,'Online now' if not inbox else 'Search conversations')
  if inbox:
   for j in range(7):
    y=125+j*96;s.rect((p,y,480-p,y+85));s.rect((p+12,y+14,p+58,y+60),s.accent,23)
    s.text(p+72,y+10,PEOPLE[(index+j)%len(PEOPLE)],19)
    s.text(p+72,y+41,r.choice(['You: That sounds good!','Can you send the link?','Meet at the cafe at six?','Thanks for the notes.','Did the order arrive?']),14,s.muted,width=300)
  else:
   phrases=[f'Hey {person}, are you free later?',f'I found {product.lower()} on sale.',f'Want to meet near {place}?','Sure, send me the details.','The article was helpful. Thanks!','Can you reply before tonight?','Yes, that works for me.','See you there!']
   r.shuffle(phrases);y=130
   for j in range(r.randrange(5,8)):
    right=(j+family)%2; x=125 if right else p; end=480-p if right else 357
    lines=textwrap.wrap(phrases[j],28);hh=24+len(lines)*25
    s.rect((x,y,end,y+hh),s.accent if right else s.card,17)
    s.text(x+13,y+12,'\n'.join(lines),17,'#ffffff' if right else s.fg)
    s.text(end-56,y+hh-15,f'{8+j}:24',10,s.muted if not right else '#edf2ff');y+=hh+17
   s.rect((p,826,480-p,880));s.text(p+15,843,r.choice(['Type a message','Message','Write a reply']),17,s.muted);s.text(416,842,'>',20,s.accent)
  s.footer('Chats        Calls        Contacts        Settings')
 elif label=='social_meme':
  s.header(r.choice(['For you','Following','Discover','Now playing']))
  s.text(p,112,person+'  @weekendnotes',17);s.text(400,112,'Follow',14,s.accent)
  s.art((p,150,480-p,465),'landscape' if family%3 else 'product')
  s.text(p+12,225,r.choice(['Weekend mood','A little joy','One more episode','Good things ahead']),27,'#ffffff',width=385)
  s.text(p,487,r.choice(JOKES),21,width=440-2*p)
  s.text(p,593,f'{r.randrange(80,980)} likes    {r.randrange(3,90)} comments',15,s.muted)
  s.rect((p,631,480-p,681));s.text(p+14,647,'Like        Comment        Share',17)
  s.text(p,710,'Replies',18);s.text(p+8,753,r.choice(['This made my day!','Sending this to the group.','The best part of the weekend.']),17,width=420)
  s.footer('Feed         Search         Create         Profile')
 elif label=='shopping_product':
  s.header(r.choice(['Shop','New arrivals','Product details','Your bag']),'Search products and brands')
  if family%3==1:
   for j in range(4):
    y=130+j*170;s.rect((p,y,480-p,y+154));s.art((p+12,y+12,p+131,y+142),'product')
    s.text(p+148,y+17,PRODUCTS[(index+j)%len(PRODUCTS)],20,width=255)
    s.text(p+148,y+75,f'${price+j*7:.2f}',23,s.accent);s.text(p+148,y+112,'Add to bag',16)
  else:
   s.art((p,126,480-p,467),'product');s.text(p,490,product,27,width=430)
   s.text(p,549,f'${price:.2f}',30,s.accent);s.text(230,560,'20% off today',17,s.muted)
   s.text(p,613,'4.7 / 5   '+str(r.randrange(20,900))+' reviews',16)
   s.text(p,657,'Choose a color       Available in stock',16,s.muted)
   s.text(p,708,'Free shipping on eligible orders',16)
  s.button(817,r.choice(['Add to cart','Add to bag','Buy now']));s.footer('Shop          Search          Bag          Account')
 elif label=='map_place':
  s.header(place,'Search nearby places')
  mapbox=(p,119,480-p,580);s.rect(mapbox,'#dbe7df',8)
  for j in range(7):
   x=p+30+j*62;s.d.line([s.xy((x,120)),s.xy((x-70,580))],fill='#ffffff',width=max(3,round(12*s.scale)))
  for j in range(6):
   y=140+j*75;s.d.line([s.xy((p,y)),s.xy((480-p,y+25))],fill='#ffffff',width=max(3,round(10*s.scale)))
  for x,y,name in [(45,182,'Maple Ave'),(200,320,'Oak Street'),(85,463,'River Road')]:s.text(x,y,name,13,'#586c65')
  s.d.line([s.xy((76,510)),s.xy((120,390)),s.xy((295,405)),s.xy((332,238))],fill='#397bd4',width=max(3,round(7*s.scale)))
  s.rect((312,212,347,247),'#cc634e',17);s.text(321,218,'P',17,'#ffffff')
  s.rect((p,603,480-p,791));s.text(p+16,621,place,24,width=400)
  s.text(p+16,668,f'4.{r.randrange(3,9)} stars   Open until 8 PM',16,s.muted)
  s.text(p+16,711,f'{r.randrange(12,800)} Maple Avenue',17);s.text(p+16,750,'About 12 minutes away',15,s.muted)
  s.button(816,r.choice(['Get directions','Save this place','Plan route']));s.footer('Explore         Saved         Updates')
 elif label=='article_reading':
  titles=['Small habits for a calmer morning','A weekend guide to your neighborhood','How to compare everyday products','Understanding a simple algorithm','A practical guide to taking better notes','Designing clear interfaces']
  s.header(r.choice(['Daily Journal','Reading list','Notebook','Learning Hub']))
  y=s.text(p,123,titles[index%len(titles)],29,width=440)
  s.text(p,y+8,'By '+person+'  |  6 min read',14,s.muted);y+=55
  if family%3!=2:s.art((p,y,480-p,y+170));y+=191
  for j in range(3+(family%2)):
   y=s.text(p,y,PARAGRAPHS[(index+j)%len(PARAGRAPHS)],17,width=440-2*p)+16
   if y>817:break
  s.footer('Back          Bookmark          Share')
 elif label=='receipt_confirmation':
  s.header(r.choice(['Order details','Payment details','Booking','Receipt']))
  s.rect((p,121,480-p,797));s.rect((205,143,275,213),s.accent,35);s.text(227,162,'OK',18,'#ffffff')
  s.text(p+30,241,r.choice(['Payment successful','Order confirmed','You are all set']),27,width=390)
  s.text(p+30,295,f'Reference: {r.randrange(100000,999999)}',15,s.muted)
  s.text(p+30,333,'2026-05-'+str(r.randrange(1,29)).zfill(2)+'  14:35',15,s.muted)
  s.text(p+30,390,r.choice([product,place,'Weekend rail ticket']),21,width=375)
  subtotal=r.randrange(10,150);tax=round(subtotal*.07,2)
  for j,(key,value) in enumerate([('Subtotal',subtotal),('Tax',tax),('Total paid',subtotal+tax)]):
   s.text(p+30,466+j*57,key,19);s.text(325,466+j*57,f'${value:.2f}',19)
  s.text(p+30,666,'Paid with card ending 0042',16,s.muted);s.text(p+30,711,'Keep this confirmation for your records.',15,width=380)
  s.button(819,r.choice(['Download receipt','View order','Add ticket to wallet']));s.footer('Home          Activity          Account')
 elif label=='app_screen':
  s.header(r.choice(['Settings','Account','Device preferences','Storage']))
  options=['Notifications','Privacy controls','Display and brightness','Connected devices','Language and region','Storage usage','Help and support']
  for j,name in enumerate(options):
   y=125+j*91;s.rect((p,y,480-p,y+78));s.rect((p+12,y+19,p+46,y+53),s.accent,9)
   s.text(p+61,y+14,name,18,width=330);s.text(p+61,y+43,r.choice(['Enabled','Manage preferences','Tap to configure','Default']),13,s.muted)
   s.text(435,y+27,'>',18,s.muted)
  s.footer('Home          Tools          Profile')
 else:
  # A full-screen wallpaper/illustration capture, without a task-specific app.
  s.art((0,42,480,s.h/s.scale),'landscape')
  if family%3==1:
   for j in range(12):
    x=r.randrange(15,430);y=r.randrange(110,800);sz=r.randrange(15,65)
    s.d.ellipse(s.xy((x,y,x+sz,y+sz)),fill=r.choice(['#e8c0bd','#aec4a5','#edd7a1','#9fb8ce']))
  if index%4==0:s.text(35,132,r.choice(['a quiet moment','color study','summer sketch','slow afternoon']),28,'#ffffff',width=400)
 return s.im


def generate(per_category=200):
 if per_category%10 or per_category<20:
  raise ValueError('per-category must be a multiple of ten, at least twenty.')
 dest=ROOT/'data/synthetic-screens';images=dest/'images';images.mkdir(parents=True,exist_ok=True)
 rows=[];seen=set();per_family=per_category//10
 for category in NAMES:
  for family in range(10):
   split='train' if family<6 else ('validation' if family<8 else 'test')
   for index in range(per_family):
    key=hashlib.sha256(f'{VERSION}/{SEED}/{category}/{family}/{index}'.encode()).hexdigest()[:20]
    path=images/(key+'.png');im=render(category,family,index);im.save(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in seen:raise ValueError('Duplicate synthetic image bytes')
    seen.add(digest)
    rows.append(dict(id=key,path=str(path.relative_to(dest)),sha256=digest,label=category,
                     split=split,group=f'{category}/family-{family}',family=family,
                     origin='synthetic_renderer',generator=VERSION,seed=SEED))
  print(f'Generated {category}: {per_category}',flush=True)
 atomic(dest/'manifest.jsonl',''.join(json.dumps(r)+'\n' for r in rows).encode())
 summary=dict(images=len(rows),per_category=per_category,counts=dict(Counter(r['label'] for r in rows)),
              splits=dict(Counter(r['split'] for r in rows)),groups=len({r['group'] for r in rows}),
              exact_duplicates=0,generator=VERSION,seed=SEED,
              limitation='Fictional code-rendered screens with generator-intent labels. Configuration families are held out, but renderer logic and phrase banks are shared. Not a real-screenshot accuracy estimate.')
 atomic(dest/'inventory.json',json.dumps(summary,indent=2).encode())
 # A balanced visual sample for checking the renderer without sharing the corpus.
 sheet=Image.new('RGB',(4*250,2*540),'#e8ecf0');d=ImageDraw.Draw(sheet)
 for j,category in enumerate(NAMES):
  im=render(category,8,j).resize((240,480))
  x=(j%4)*250+5;y=(j//4)*540+37
  sheet.paste(im,(x,y));d.text((x,y-27),NAMES[category],font=ImageFont.load_default(size=13),fill='#142b3a')
 preview=ROOT/'reports/feature-study';preview.mkdir(parents=True,exist_ok=True);sheet.save(preview/'synthetic-preview.jpg',quality=90)
 print(json.dumps(summary,indent=2))
 return rows


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--per-category',type=int,default=200)
 generate(parser.parse_args().per_category)
