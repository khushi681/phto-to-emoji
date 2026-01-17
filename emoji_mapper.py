import cv2

emoji_dict = {
    "angry": "emojis/angry.png",
    "disgust": "emojis/disgust.png",
    "fear": "emojis/fear.png",
    "happy": "emojis/happy.png",
    "neutral": "emojis/neutral.png",
    "sad": "emojis/sad.png",
    "surprise": "emojis/surprise.png"
}

def get_emoji(emotion):
    emoji_path = emoji_dict.get(emotion)
    if emoji_path:
        return cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    return None
