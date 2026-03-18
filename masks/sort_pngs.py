import os
import shutil

# Target directory
target_dir = "/Users/scohn/code/AceStep15-gradio2comfy/masks/donotcommit/pngs"

# Category mapping (Keywords to Category Folder)
CATEGORIES = {
    "Audio_Music": [
        "acoustic", "album", "beatport", "boombox", "cassette", "cd", "drum", "drums", "equalizer", 
        "guitar", "instrument", "jingle", "jukebox", "metronome", "mic", "piano", "podcast", 
        "radio", "rap", "rapper", "recording", "rhythm", "rock", "soundwave", "vinyl", 
        "tambourine", "bagpipes", "banjo", "bugle", "bugles", "dizi", "dulcimer", "french-horn", 
        "harmonica", "maraca", "percussion", "heavymetal", "audio", "lp-scratch", "speaker", 
        "sound", "indie", "pop", "live", "beat", "voice", "algorhythm", "boombox", "concert", 
        "heavy-metal", "karaoke", "music", "record", "studio", "synth", "tape", "tune", "volume",
        "bullhorn", "two-drumsticks", "voice", "audio", "recording", "radio", "speaker"
    ],
    "Dance_Performance": [
        "ballerinas", "ballet", "cheerleader", "chinese-fan", "dance", "dancer", "dazzle", 
        "flamenco", "gymnast", "gymnastics", "hula-hoop", "ice-dance", "jumping-dancer", 
        "just-dance", "perform", "skate", "choreography", "stage", "theather", "tragedy",
        "olympic", "skate"
    ],
    "Western_Country": [
        "cowboy", "cowgirl", "western", "lasso", "sheriff", "cactus", "watertower", "cork-hat",
        "western-digital", "cowboy-hat", "cowboy-boot", "roots", "hat", "leather", "high-boots"
    ],
    "Food_Drink": [
        "avocado", "bacon", "beer", "bottle", "bread", "candy", "carrot", "cherry", "chicken", 
        "cocktail", "coffee", "donut", "egg", "eggplant", "fries", "hamburger", "hotdog", 
        "icecream", "lemon", "milk", "orange", "onion", "pasta", "pepperoni", "pizza", 
        "pretzel", "steak", "strawberry", "taco", "tea", "watermelon", "wine", "muffin", 
        "croisant", "acorn", "bowl", "fork", "spoon", "knife", "mug", "baking", "chef", 
        "dinner", "drink", "food", "grapes", "honeycomb", "jar", "juice", "kitchen", 
        "mustard", "salt", "sauce", "vegetable", "cherry", "coffeebean", "foodtray", 
        "kiwifruit", "lollypop", "muffin", "pasta", "restaurantmenu", "sizzle", "spaghetti", 
        "taco", "tallglass", "toast", "turnip", "wineglass", "carrot", "candy", "drink", 
        "fries", "hamburger", "pizza", "taco", "wine"
    ],
    "Tech_Development": [
        "android", "apple", "blackberry", "ios", "linux", "windows", "ubuntu", "fedora", 
        "archlinux", "elementaryos", "google", "facebook", "instagram", "twitter", "youtube", 
        "skype", "whatsapp", "dropbox", "github", "bitbucket", "stackoverflow", "reddit", 
        "linkedin", "gmail", "outlook", "ie", "safari", "chrome", "firefox", "opera", 
        "nodejs", "python", "java", "php", "ruby", "mysql", "mongodb", "nosql", "cms", 
        "wordpress", "magento", "joomla", "drupal", "cloud", "compile", "code", "commit", 
        "database", "db", "ethernet", "ftp", "harddisk", "hdd", "hosting", "server", "ssl", 
        "versions", "wacom", "web", "website", "wifi", "binary", "browser", "computer", 
        "cpu", "data", "develop", "digit", "email", "hardware", "internet", "laptop", 
        "network", "os", "programming", "software", "tech", "analytics", "antivirus", 
        "apache", "aperture", "apps", "authentication", "awstats", "basecamp", "bigace", 
        "bitcoin", "boxbilling", "boxtrapper", "cgi", "cloudhosting", "cmsmadesimple", 
        "codeigniter", "collabtive", "connectedpc", "cssthree", "dedicatedserver", 
        "dotclear", "elgg", "extjs", "facetimevideo", "fantastico", "feedly", "fengoffice", 
        "filemanager", "firewire", "fitocracy", "flickr", "floppy", "forrst", "foursquare", 
        "future", "grails", "greenhosting", "hangout", "hangouts", "hdtv", "hp", "ie9", 
        "java", "limesurvey", "livejournal", "macpro", "macro", "mahara", "mambo", 
        "managedhosting", "mongodb", "mootools", "mybb", "mysql", "neofourj", "nexus", 
        "nodejs", "nosql", "opencart", "openclassifieds", "openid", "opensource", "osclass", 
        "oscommerce", "panoramio", "photobucket", "picasa", "pimcore", "pivotx", "pligg", 
        "plogger", "prestashop", "pyrocms", "python", "raspberrypi", "redirect", 
        "resellerhosting", "roundcube", "rss", "ruby", "safari", "sencha", "sidu", 
        "simplepie", "skitch", "skype", "smarty", "socialnetwork", "software", "soundcloud", 
        "spotify", "sslmanager", "stackoverflow", "subrion", "svg", "tampermonkey", 
        "taskfreak", "technorati", "tethering", "textfield", "tomatocart", "twitter", 
        "typothree", "ubuntu", "viadeo", "viber", "webcam", "webhostinghub", "webplatform", 
        "website", "whatsapp", "whmcs", "windows", "wordpress", "yiiframework", "youtube", 
        "zikula", "ads", "arch", "briefcase", "controller", "dns", 
        "install", "minecraft", "plugin", "project", "safety", "switch", "system", "usb",
        "windowseight", "zoomin", "acw", "aef", "asl", "aumade", "burstmode", "cctv", 
        "currency", "currents", "dashboard", "designcontest", "digg", "diskspace", 
        "eur", "euro", "finance", "finder", "gps", "hotlink", "ingress", "ipad", "ipod", 
        "lan", "launch", "law", "layout", "microsd", "mybb", "newtab", "notes", "nucleus", 
        "pagebreak", "pc", "pied-piper", "plug", "post", "raspberry", "reliability", 
        "residentevil", "router", "security", "timeline", "timer", "unpackarchive", 
        "usb", "visa", "websitebuilder"
    ],
    "Nature_Outdoors_Space_Science": [
        "anchor", "beach", "birdhouse", "branch", "butterfly", "campfire", "city", "clover", 
        "fence", "flower", "flowerpot", "forest", "fountain", "island", "leaf", "lily", 
        "mountains", "mushroom", "palm", "sea", "sunrise", "sunset", "tree", "windmill", 
        "garden", "landscape", "nature", "outdoor", "park", "plant", "river", "sky", "water", 
        "bamboo", "birdhouse", "branch", "butterfly", "fireplace", "flame", "flower", 
        "flowerpot", "forest", "fountain", "island", "leaf", "lighthouse", "lily", 
        "mountains", "mushroom", "palm", "reproduction", "sea", "sunrise", "sunset", "tree", 
        "windmill", "world", "alienship", "alienware", "asteroid", "astronaut", "atom", 
        "constellation", "deathstar", "dna", "galaxy", "meteor", "meteorite", "observatory", 
        "planet", "rocket", "satellite", "shuttle", "solarsystem", "sputnik", "telescope", 
        "black-hole", "spaceship", "universe", "alien", "moon", "sun", "star", "physics", 
        "science", "biohazard", "chemistry", "fallingstar", "flask", "globe", "lab", "lava", 
        "lightning", "microscope", "molecule", "nuclear", "radioactive", "virus", "sputnik",
        "cloud", "rain", "snow", "storm", "thunder", "hail"
    ],
    "Sports_Games": [
        "baseball", "basketball", "bowling", "cricket", "curling", "dice", "football", "golf", 
        "hockey", "joystick", "olympic", "pacman", "pingpong", "playstation", "pokemon", 
        "racquet", "stadium", "tennis", "usfootball", "zelda", "controller", "game", "chess", 
        "bishop", "king", "queen", "rook", "pawn", "play", "sport", "team", "tournament", 
        "joystickarcade", "joystickatari", "nintendods", "quake", "warcraft", "oneup", 
        "boxing", "boardgame", "gameboy", "controller"
    ],
    "Transport_Travel": [
        "automobile", "boat", "bus", "car", "helicopter", "plane", "railroad", "ship", 
        "submarine", "taxi", "train", "truck", "freeway", "ambulance", "bike", "bicycle", 
        "motorcycle", "highway", "road", "traffic", "transport", "vehicle", "gasstation", 
        "intersection", "landing", "metro", "parkingmeter", "railtunnel", "roadmap", 
        "roadtunnel", "route", "takeoff", "trailor", "travel"
    ],
    "Objects_Misc": [
        "bag", "basket", "bed", "binoculars", "box", "briefcase", "camera", "chair", "clock", 
        "compass", "crown", "gift", "hammer", "ironman", "lamp", "mickeymouse", "pacifier", 
        "paint", "pen", "phone", "printer", "puzzle", "scissors", "screwdriver", "shirt", 
        "shoes", "skull", "spiderman", "suitcase", "tools", "umbrella", "watch", "wrench", 
        "battery", "bulb", "candle", "clipboard", "cigar", "cigarette", "extinguisher", 
        "flashlight", "glasses", "glue", "iron", "kettle", "ladder", "megaphone", "mirror", 
        "money", "needle", "newspaper", "notebook", "ruler", "safe", "soap", "stapler", 
        "stroller", "tablet", "ticket", "tie", "wallet", "apparel", "clothing", "fashion", 
        "obj", "object", "tool", "wear", "acorn", "axe", "badge", "balloon", "bat", 
        "bathtub", "belt", "bomb", "book", "bowtie", "brain", "broom", "brush", "bucket", 
        "bullet", "canister", "cannon", "ceilinglight", "certificatethree", "classic-sewing", 
        "clutch", "coins", "construction", "cooling", "corckscrew", "crayon", "cube", 
        "dagger", "desklamp", "diamond", "diamonds", "drawer", "dry", "earbuds", "elevator", 
        "fan", "film", "filmstrip", "firewire", "firstaid", "flaskfull", "fossil", "fridge", 
        "gavel", "gear", "ghost", "ghostface", "glass", "glow", "graduation", "grave", 
        "grip", "gun", "handcuffs", "hanger", "hoddie", "hydrant", "ink", "inkpen", "inkscape", 
        "ironman", "jawbreaker", "jigsaw", "kettle", "kidney", "leather", "legacyfilemanager", 
        "lego", "lens", "lifepreserver", "lips", "lipstick", "liver", "lungs", "macro", 
        "magic", "magnet", "manillaenvelope", "mexican-skull", "microwave", "mimetype", 
        "minecraft", "molecule", "monitor", "monstra", "moustache", "movieclapper", 
        "moviereel", "multiply", "nail", "noview", "nut", "officechair", "pacifier", 
        "package", "paint", "pallete", "panties", "paper", "parentheses", "parthenon", 
        "patch", "pawn", "pear", "pebble", "perfume", "photo", "photosphere", "picasa", 
        "pickaxe", "picker", "piechart", "pipe", "pitchfork", "place", "plane", "plaque", 
        "play", "plugin", "plumbing", "podium", "pointer", "police", "polygonlasso", 
        "poop", "popcorn", "present", "presentation", "preview", "price", "print", 
        "programok", "projectcompare", "projector", "projectpier", "protecteddirectory", 
        "pullrequest", "pumpjack", "punisher", "punk", "purse", "puzzle", "quake", "razor", 
        "resistor", "restart", "restricted", "retweet", "rewind", "rim-", "ring", 
        "robocop", "roots", "rorschach", "rotate", "rubberstamp", "safety", "sale", 
        "saurus", "scales", "screw", "script", "sencha", "sendtoback", "sextant", "sheep", 
        "shopping", "shovel", "shredder", "skitch", "skull", "smallgear", "smarty", 
        "smile", "snaptogrid", "snooze", "snowman", "solarpanel", "spaceinvaders", 
        "sparkle", "spawn", "speech", "speed", "sphere", "spider", "spiderweb", "splash", 
        "split", "spock", "stamp", "sticker", "stiletto", "stomach", "stopsign", 
        "stopwatch", "store", "storm", "student", "style", "surgical", "suitcase", "sword", 
        "tablet", "tape", "temp", "temple", "texture", "ticket", "tie", "toiletbrush", 
        "tooth", "toothbrush", "tophat", "torch", "torigate", "tornado", "trash", 
        "trophy", "usb", "usd", "vaultthree", "vector", "vendetta", "vial", "video", 
        "view", "wallet", "wand", "washer", "weightscale", "wetfloor", "wheel", 
        "whistle", "wizard", "workshirt", "wrench", "xmen", "xoops", "bait", "baloon",
        "breakable", "broomstick", "cameraflash", "candycane", "cart", "charging",
        "climb", "glow", "high-boots", "lego", "pamela", "paperboat", "papercutter",
        "pitchfork", "rim", "rocker", "safetygoggles", "safetypin", "shirtbuttonthree",
        "shoppingbag", "shoppingcart", "smallgear", "sparkle", "splash", "stacks",
        "stamp", "sticker", "stiletto", "temp", "three-fingers", "threed", "two-drumsticks",
        "two-fingers", "usbflash", "usbplug", "vaultthree", "vial", "warningsign",
        "wheelchair", "blind", "bootleg", "bow", "bug", "caligraphy", "camcorder", "cc", 
        "cell", "crackedegg", "craft", "creeper", "danger", "dart", "doghouse", "energy", 
        "escalator", "fire", "fort", "greekcolumn", "hail", "harrypotter", "hot", "house", 
        "impaired", "manualshift", "marvin", "middlefinger", "mouse", "mypictures", 
        "myvideos", "panorama", "red-hot", "rudder", "spray", "tank", "tide", "tv", 
        "twirl", "yelp", "yinyang"
    ],
    "Symbols_UI_Interface": [
        "arrow", "bell", "bookmark", "checked", "chevron", "circle", "cog", "comment", "cursor", 
        "delete", "eye", "favorite", "flag", "folder", "heart", "info", "key", "link", 
        "location", "lock", "magnifier", "map", "marker", "notification", "pencil", "plus", 
        "question", "refresh", "rss", "search", "settings", "share", "shield", "smile", 
        "star", "tag", "trash", "user", "warning", "bolt", "cross", "emergency", "peace", 
        "recycle", "alert", "check", "close", "home", "icon", "list", "menu", "power", 
        "signal", "ui", "zoom", "alarm", "asterisk", "blankstare", "ccw", "chat", 
        "connected", "contract", "day", "direction", "directions", "disconnect", "dislike", 
        "distribute", "dns", "dollar", "donotdisturb", "doubletap", "drop", "empty", 
        "event", "exchange", "fave", "filter", "full", "half", "history", "hourglass", 
        "idea", "inbox", "ok", "open", "pointer", "plus", "quote", "redo", "refresh", 
        "restricted", "rewind", "rotate", "sign", "undo", "bubbles", "docs", "graph", 
        "layers", "line", "pawn", "pin", "play", "stopwatch", "target", "thinking", "slash",
        "status", "marker", "map", "compass", "bubble", "check", "clip", "clubs", 
        "colors", "community", "cone", "connect", "createfolder", "cut", "diagram", 
        "die", "elipse", "fb", "forward", "green", "hand", "happy", "help", "hexagon", 
        "innerborders", "life", "lighton", "like", "loaction2", "marker", "merge", 
        "off", "outerborders", "rectangle", "refused", "roundrectangle", "run", 
        "spades", "square", "stocks", "survey", "talke", "tap", "tint", "zoom"
    ],
    "People_Characters": [
        "baby", "batman", "captainamerica", "charliechaplin", "chef", "darthvader", 
        "director", "drmanhattan", "female", "ghostface", "ironman", "jason", "male", 
        "mickeymouse", "pamela", "police", "punisher", "punk", "robocop", "rorschach", 
        "spiderman", "spock", "student", "superman", "vendetta", "viking", "woman", 
        "women"
    ],
    "Holiday_Culture_Religion": [
        "birthday", "christmas", "halloween", "easter", "party", "event", "gift", "present", 
        "raceflag", "chinese-gong", "kathakali", "leprechaun", "mayan", "mosque", "temple", 
        "islam", "christiancross", "davidstar", "parthenon", "torigate", 
        "jiangxi", "mun-", "paydarymelli", "lalibela", "christmasstree", "church",
        "mayanpyramid", "mexican"
    ],
    "Emotions_Faces": [
        "blankstare", "emo-", "emoji", "smile", "sad", "laugh", "grin", "confused", 
        "surprise", "sorry", "cry", "tongue", "kiss", "lips", "commentsmiley",
        "miniangry", "miniconfused", "minigrin", "minilaugh", "minisad", "minismile",
        "minitongue"
    ]
}

def get_category(filename):
    name = filename.lower()
    
    # Try startswith matching (aggressive prefix check)
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if name.startswith(kw):
                return cat
                
    # Try containment check for distinct word parts
    parts = name.replace('.png', '').split('-')
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in parts:
                return cat
                
    return "Uncategorized"

def sort_files():
    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        return

    # Gather all PNGs recurvisely
    all_files = []
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith('.png'):
                all_files.append(os.path.join(root, f))
    
    count = 0
    moved_counts = {}

    for filepath in all_files:
        filename = os.path.basename(filepath)
        category = get_category(filename)
        dest_dir = os.path.join(target_dir, category)
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            
        dest_path = os.path.join(dest_dir, filename)
        
        # Don't move if it's already in the right place
        if os.path.abspath(filepath) == os.path.abspath(dest_path):
            continue
            
        shutil.move(filepath, dest_path)
        count += 1
        moved_counts[category] = moved_counts.get(category, 0) + 1

    # Cleanup empty directories
    for d in os.listdir(target_dir):
        dir_path = os.path.join(target_dir, d)
        if os.path.isdir(dir_path) and d != "Uncategorized":
            if not os.listdir(dir_path):
                 os.rmdir(dir_path)

    print(f"Final re-sort: {count} files moved.")
    for cat, c in sorted(moved_counts.items()):
        print(f"  {cat}: {c}")

if __name__ == "__main__":
    sort_files()
