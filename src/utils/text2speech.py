import os
import tempfile
# from TTS.api import TTS
import requests
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
import time

OPENAI_API_KEY = "sk-bA4N4H8xDy5uurP6y5o4EBXHi0ewyT3lzVvW2kepzKPtYDlP"
OPENAI_BASE_URL = "https://api.closeai-asia.com/v1"


TTS_MODEL = "tts-1"
TTS_URL = "https://api.closeai-proxy.xyz/v1/audio/speech"


class TTSTalker():
    """文本转语音"""

    def __init__(self) -> None:
        model_name = TTS().list_models()[0]
        self.tts = TTS(model_name)

    def test(self, text, language='en'):

        tempf = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=('.'+'wav'),
        )

        self.tts.tts_to_file(
            text, speaker=self.tts.speakers[0], language=language, file_path=tempf.name)

        return tempf.name


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def tts_request(text: str, save_dir: str = './audio_cache/', url: str = TTS_URL, model: str = TTS_MODEL):
    """
    https://platform.openai.com/docs/guides/text-to-speech

    text to speech here.

    Parameters:
        text: str, 待转换的文本
    Returns:
        path: str, 保存音频的路径
    """

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }

    query = {
        "model": model,
        "input": text,
        "voice": "alloy",
        "response_format": "wav",  # mp3
        "speed": 1,
    }

    try:
        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(query)
        )
        response.raise_for_status()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        time_stamp = time.strftime("%Y.%m.%d-%H.%M.%S")

        audio_path = os.path.join(save_dir, 'audio_' + time_stamp + '.wav')
        with open(audio_path, 'wb') as f:
            f.write(response.content)

        print(f"\n回复音频保存文件路径: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Unable to convert text to speech")
        print(f"Exception: {e}")
        return e


class TTS_OPENAI():
    def test(self, text, language='zh'):
        if not text:
            return None

        print(f"text is {text}.")

        try:
            audio_path = tts_request(text)
            return audio_path
        except Exception as e:
            print(f"Unable to convert text to speech.")
            print(f"EXception {e}")
            return None
