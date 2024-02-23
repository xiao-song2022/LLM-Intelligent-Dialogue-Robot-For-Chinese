import tkinter as tk
from tkinter import scrolledtext
import threading
import pyaudio
import wave
import pygame
from transformers import AutoTokenizer, AutoModel
import whisper
import os
import torch
import OpenVoice.se_extractor
from OpenVoice.api import BaseSpeakerTTS, ToneColorConverter



class AudioRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("语音对话程序")

        self.recording = False
        self.history = []

        self.record_button = tk.Button(self.root, text="开始录音", command=self.toggle_recording)
        self.record_button.pack(pady=10)

        self.chat_box = scrolledtext.ScrolledText(self.root, width=50, height=20)
        self.chat_box.pack(pady=10)

        self.exit_button = tk.Button(self.root, text="退出", command=self.exit_program)
        self.exit_button.pack(pady=10)

        # 初始化模型和引擎
        self.tokenizer = AutoTokenizer.from_pretrained("E:\chat\ChatGLM-6B-main\dataroot\models\THUDM\chatglm-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("E:\chat\ChatGLM-6B-main\dataroot\models\THUDM\chatglm-6b", trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

        # self.engine = pyttsx3.init()

        device="cuda:0" if torch.cuda.is_available() else "cpu"
        ckpt_base = 'OpenVoice/checkpoints/base_speakers/ZH'
        self.base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
        self.base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')
    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.record_button.config(text="停止录音")
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.start()
        else:
            self.recording = False
            self.record_button.config(text="开始录音")

    def record_audio(self):
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 2
        fs = 44100
        filename = "audio.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []

        while self.recording:
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        result = self.transcribe_audio(filename)

        self.chat_box.insert(tk.END, "You: {}\n".format(result))

        response, self.history = self.chat_with_model(result, history=self.history)
        self.chat_box.insert(tk.END, "AI: {}\n".format(response))

        # 播放AI回答的音频
        self.save_to_audio_file(response, "ai_response.mp3")
        self.play_audio("ai_response.mp3")

    def transcribe_audio(self, audio_filename):
        model = whisper.load_model("base")
        result = model.transcribe(audio_filename)
        return result["text"]

    def chat_with_model(self, input_text, history):
        response, new_history = self.model.chat(self.tokenizer, input_text, history=history)
        return response, new_history

    def save_to_audio_file(self, text, audio_filename):
        self.base_speaker_tts.tts(text, audio_filename, speaker='default', language='Chinese', speed=1.0)

    def play_audio(self, audio_filename):
        pygame.init()
        sound = pygame.mixer.Sound(audio_filename)
        sound.play()

    def exit_program(self):
        self.recording = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.run()