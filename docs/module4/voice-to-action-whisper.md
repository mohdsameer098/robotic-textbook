# Module 4, Chapter 2 - Voice-to-Action with OpenAI Whisper

## Introduction to Speech Recognition for Robotics

Voice interaction is the most natural form of human-robot communication. Instead of typing commands or using controllers, users can simply speak to their humanoid robot: "Go to the kitchen and bring me a water bottle." This requires robust **Automatic Speech Recognition (ASR)** that can:

- Work in noisy environments (robot motors, ambient sounds)
- Handle various accents and speaking styles
- Operate in real-time with low latency
- Run efficiently on edge devices (Jetson)
- Support multiple languages

**OpenAI Whisper** has emerged as the gold standard for speech recognition, offering state-of-the-art accuracy across 99 languages while being open-source and deployable on edge hardware.

## What is OpenAI Whisper?

Whisper is a neural network trained on 680,000 hours of multilingual and multitask supervised data collected from the web. Unlike traditional ASR systems, Whisper is:

### Key Features

1. **Multilingual**: Supports 99 languages, including:
   - English, Spanish, French, German, Chinese, Japanese
   - Urdu, Hindi, Arabic, Russian, Portuguese
   - Even low-resource languages like Swahili, Punjabi

2. **Robust to Noise**: Trained on real-world data with background noise, accents, and audio distortions

3. **Multi-task**: Can perform:
   - Speech recognition (transcription)
   - Speech translation (any language → English)
   - Language identification
   - Voice activity detection

4. **Zero-shot Transfer**: Works out-of-the-box without fine-tuning

5. **Open Source**: Free to use, modify, and deploy

### Model Sizes

Whisper comes in five sizes, trading accuracy for speed:

| Model | Parameters | English-only | Multilingual | Relative Speed | VRAM |
|-------|-----------|--------------|--------------|----------------|------|
| tiny | 39M | ✓ | ✓ | ~32x | ~1 GB |
| base | 74M | ✓ | ✓ | ~16x | ~1 GB |
| small | 244M | ✓ | ✓ | ~6x | ~2 GB |
| medium | 769M | ✓ | ✓ | ~2x | ~5 GB |
| large | 1550M | ✗ | ✓ | 1x | ~10 GB |

**For humanoid robots:**
- **Development/Testing**: Use `small` or `medium` on workstation
- **Edge Deployment**: Use `tiny` or `base` on Jetson Orin Nano
- **Cloud Processing**: Use `large` for maximum accuracy

## Installing Whisper

### Option 1: OpenAI's Official Implementation

```bash
# Install Whisper and dependencies
pip install openai-whisper

# Or install with CUDA support for GPU acceleration
pip install openai-whisper torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ffmpeg (required for audio processing)
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

### Option 2: Faster Whisper (Recommended for Production)

`faster-whisper` is a reimplementation using CTranslate2, offering 4x speedup with lower memory usage:

```bash
pip install faster-whisper
```

### Option 3: Whisper.cpp (For Edge Devices)

C++ implementation optimized for CPU and edge devices:

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make

# Download a model (e.g., base)
bash ./models/download-ggml-model.sh base
```

### Verify Installation

```python
import whisper

# Load model
model = whisper.load_model("base")

# Test transcription
result = model.transcribe("audio.mp3")
print(result["text"])
```

## Basic Whisper Usage

### Transcribing Audio Files

```python
import whisper

# Load model
model = whisper.load_model("base")

# Transcribe audio file
result = model.transcribe("robot_command.wav")

print("Transcription:", result["text"])
print("Language detected:", result["language"])
print("Segments:", result["segments"])
```

**Output:**
```
Transcription: Go to the kitchen and bring me a glass of water
Language detected: en
Segments: [
    {'start': 0.0, 'end': 2.5, 'text': 'Go to the kitchen'},
    {'start': 2.5, 'end': 5.0, 'text': 'and bring me a glass of water'}
]
```

### Transcribing with Options

```python
result = model.transcribe(
    "audio.wav",
    language="en",              # Force English (faster than auto-detect)
    task="transcribe",          # or "translate" to convert to English
    temperature=0.0,            # Deterministic output
    beam_size=5,                # Beam search size (higher = more accurate, slower)
    best_of=5,                  # Number of candidates
    word_timestamps=True,       # Get timestamps for each word
    initial_prompt="Robot commands:" # Context hint for better accuracy
)

# Access word-level timestamps
for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']} ({word['start']:.2f}s - {word['end']:.2f}s)")
```

**Output:**
```
Go (0.00s - 0.20s)
to (0.20s - 0.30s)
the (0.30s - 0.45s)
kitchen (0.45s - 0.90s)
and (0.90s - 1.05s)
bring (1.05s - 1.35s)
...
```

## Real-Time Speech Recognition

For robot control, we need real-time transcription as the user speaks, not after they finish.

### Using PyAudio for Microphone Input

```python
import pyaudio
import wave
import whisper
import numpy as np

class RealtimeWhisper:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.audio = pyaudio.PyAudio()
        
        # Audio stream configuration
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper expects 16kHz
        
        # Voice activity detection threshold
        self.SILENCE_THRESHOLD = 500
        self.SILENCE_DURATION = 1.5  # seconds
        
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("Recording...")
        frames = []
        
        for _ in range(0, int(self.RATE / self.CHUNK * duration)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        
        print("Recording finished")
        stream.stop_stream()
        stream.close()
        
        return b''.join(frames)
    
    def detect_speech(self, audio_data):
        """Simple voice activity detection"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        volume = np.abs(audio_array).mean()
        return volume > self.SILENCE_THRESHOLD
    
    def record_until_silence(self):
        """Record until user stops speaking"""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("Listening... (speak now)")
        frames = []
        silent_chunks = 0
        is_speaking = False
        
        while True:
            data = stream.read(self.CHUNK)
            frames.append(data)
            
            if self.detect_speech(data):
                is_speaking = True
                silent_chunks = 0
            elif is_speaking:
                silent_chunks += 1
                
                # Stop after silence duration
                silence_chunks_threshold = int(
                    self.SILENCE_DURATION * self.RATE / self.CHUNK
                )
                if silent_chunks > silence_chunks_threshold:
                    break
        
        print("Speech ended")
        stream.stop_stream()
        stream.close()
        
        return b''.join(frames)
    
    def transcribe_audio(self, audio_data):
        """Transcribe raw audio bytes"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Transcribe
        result = self.model.transcribe(
            audio_float,
            language="en",
            fp16=False  # Use FP32 for CPU
        )
        
        return result["text"]
    
    def listen_and_transcribe(self):
        """Main loop: listen → transcribe → return text"""
        audio_data = self.record_until_silence()
        text = self.transcribe_audio(audio_data)
        return text
    
    def cleanup(self):
        """Clean up resources"""
        self.audio.terminate()

# Usage
asr = RealtimeWhisper(model_size="base")

while True:
    command = asr.listen_and_transcribe()
    print(f"You said: {command}")
    
    if "stop listening" in command.lower():
        break

asr.cleanup()
```

### Optimized Real-Time with Faster-Whisper

```python
from faster_whisper import WhisperModel

class FastRealtimeWhisper:
    def __init__(self, model_size="base"):
        # Use GPU if available
        self.model = WhisperModel(
            model_size,
            device="cuda",  # or "cpu"
            compute_type="float16"  # or "int8" for even faster
        )
        
    def transcribe_audio(self, audio_path):
        """Faster transcription"""
        segments, info = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            vad_filter=True,  # Built-in voice activity detection
            vad_parameters=dict(
                min_silence_duration_ms=500
            )
        )
        
        # Collect all segments
        text = " ".join([segment.text for segment in segments])
        return text

# Usage
asr = FastRealtimeWhisper("base")
text = asr.transcribe_audio("command.wav")
print(text)
```

## Integrating Whisper with ROS 2

To use Whisper in a ROS 2 robot system:

### Create a Whisper ROS 2 Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import pyaudio
import numpy as np

class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_node')
        
        # Publisher for transcribed commands
        self.command_publisher = self.create_publisher(
            String,
            'voice_commands',
            10
        )
        
        # Load Whisper model
        self.get_logger().info("Loading Whisper model...")
        self.model = whisper.load_model("base")
        self.get_logger().info("Whisper model loaded")
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.CHUNK = 1024
        self.RATE = 16000
        
        # Start listening
        self.timer = self.create_timer(0.1, self.listen_callback)
        
        self.is_recording = False
        self.frames = []
        self.silent_chunks = 0
        
    def listen_callback(self):
        """Continuously listen for voice commands"""
        if not self.is_recording:
            self.start_recording()
        
        # Read audio chunk
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            self.frames.append(data)
            
            # Check if speech ended
            if self.is_silent(data):
                self.silent_chunks += 1
                if self.silent_chunks > 15:  # ~1.5 seconds
                    self.process_recording()
                    self.reset_recording()
            else:
                self.silent_chunks = 0
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            self.get_logger().error(f"Audio error: {e}")
    
    def is_silent(self, audio_data):
        """Check if audio chunk is silent"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        volume = np.abs(audio_array).mean()
        return volume < 500
    
    def start_recording(self):
        """Start new recording session"""
        self.is_recording = True
        self.frames = []
        self.silent_chunks = 0
    
    def reset_recording(self):
        """Reset recording state"""
        self.is_recording = False
        self.frames = []
        self.silent_chunks = 0
    
    def process_recording(self):
        """Transcribe recorded audio and publish"""
        if len(self.frames) < 10:  # Too short
            return
        
        try:
            # Convert to numpy array
            audio_data = b''.join(self.frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe
            self.get_logger().info("Transcribing...")
            result = self.model.transcribe(audio_float, language="en")
            text = result["text"].strip()
            
            if text:
                self.get_logger().info(f"Command: {text}")
                
                # Publish command
                msg = String()
                msg.data = text
                self.command_publisher.publish(msg)
        
        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
    
    def destroy_node(self):
        """Cleanup"""
        self.audio.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhisperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### ROS 2 Launch File

```python
# launch/whisper_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='voice_control',
            executable='whisper_node',
            name='whisper_node',
            output='screen',
            parameters=[{
                'model_size': 'base',
                'language': 'en'
            }]
        )
    ])
```

### Command Handler Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CommandHandler(Node):
    def __init__(self):
        super().__init__('command_handler')
        
        # Subscribe to voice commands
        self.subscription = self.create_subscription(
            String,
            'voice_commands',
            self.command_callback,
            10
        )
        
    def command_callback(self, msg):
        """Process voice commands"""
        command = msg.data.lower()
        self.get_logger().info(f"Received command: {command}")
        
        # Parse command and execute action
        if "move forward" in command or "go forward" in command:
            self.execute_move_forward()
        elif "turn left" in command:
            self.execute_turn_left()
        elif "turn right" in command:
            self.execute_turn_right()
        elif "stop" in command:
            self.execute_stop()
        elif "pick up" in command:
            object_name = self.extract_object(command)
            self.execute_pick(object_name)
        else:
            self.get_logger().warn(f"Unknown command: {command}")
    
    def execute_move_forward(self):
        self.get_logger().info("Executing: Move forward")
        # Publish velocity command to robot
        
    def execute_turn_left(self):
        self.get_logger().info("Executing: Turn left")
        
    def execute_pick(self, object_name):
        self.get_logger().info(f"Executing: Pick up {object_name}")
        
    def extract_object(self, command):
        """Extract object name from command"""
        # Simple extraction (in practice, use NLP)
        words = command.split()
        if "pick up the" in command:
            idx = words.index("the") + 1
            return " ".join(words[idx:])
        return "unknown"

def main(args=None):
    rclpy.init(args=args)
    node = CommandHandler()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Advanced Techniques

### 1. Wake Word Detection

Don't transcribe continuously—only listen when user says "Hey Robot":

```python
import pvporcupine

class WakeWordDetector:
    def __init__(self, wake_word="hey robot"):
        # Porcupine wake word detection
        self.porcupine = pvporcupine.create(
            access_key="YOUR_PICOVOICE_KEY",
            keywords=[wake_word]
        )
        
        self.audio = pyaudio.PyAudio()
        self.whisper_model = whisper.load_model("base")
        
    def listen_for_wake_word(self):
        """Wait for wake word, then transcribe command"""
        stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        
        print("Waiting for wake word...")
        
        while True:
            pcm = stream.read(self.porcupine.frame_length)
            pcm = np.frombuffer(pcm, dtype=np.int16)
            
            keyword_index = self.porcupine.process(pcm)
            
            if keyword_index >= 0:
                print("Wake word detected! Listening for command...")
                stream.stop_stream()
                stream.close()
                
                # Now record and transcribe command
                command = self.record_and_transcribe()
                return command
    
    def record_and_transcribe(self):
        # Implementation from earlier examples
        pass
```

### 2. Language-Specific Optimization

For Urdu support (relevant for Pakistan):

```python
# Transcribe in Urdu
result = model.transcribe(
    "audio.wav",
    language="ur",  # Urdu
    task="transcribe"
)

print(result["text"])  # Output in Urdu script
```

### 3. Streaming Transcription

For ultra-low latency, use streaming mode:

```python
from faster_whisper import WhisperModel
import sounddevice as sd

class StreamingWhisper:
    def __init__(self):
        self.model = WhisperModel("base", device="cuda")
        self.buffer = []
        self.SAMPLE_RATE = 16000
        
    def audio_callback(self, indata, frames, time, status):
        """Called for each audio chunk"""
        self.buffer.append(indata.copy())
        
        # Process every 3 seconds
        if len(self.buffer) >= 3 * self.SAMPLE_RATE / frames:
            audio = np.concatenate(self.buffer)
            self.buffer = []
            
            # Transcribe chunk
            segments, _ = self.model.transcribe(audio, language="en")
            for segment in segments:
                print(f"[{segment.start:.1f}s] {segment.text}")
    
    def start_streaming(self):
        """Start streaming transcription"""
        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.SAMPLE_RATE
        ):
            print("Streaming... Press Ctrl+C to stop")
            sd.sleep(1000000)  # Stream indefinitely
```

## Deployment on Jetson Orin Nano

### Optimizing for Edge Devices

```bash
# Install optimized PyTorch
pip3 install torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v511

# Use quantized models
pip install faster-whisper

# Use int8 quantization for 2-3x speedup
model = WhisperModel("base", device="cuda", compute_type="int8")
```

### Benchmarking on Jetson

```python
import time

def benchmark_whisper(model_size, audio_file, num_runs=10):
    model = WhisperModel(model_size, device="cuda", compute_type="int8")
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        segments, _ = model.transcribe(audio_file)
        list(segments)  # Force evaluation
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    print(f"{model_size}: {avg_time:.2f}s average")

# Test all model sizes
for size in ["tiny", "base", "small"]:
    benchmark_whisper(size, "test_audio.wav")
```

**Expected Results on Jetson Orin Nano:**
- tiny: ~0.5s (real-time capable)
- base: ~1.2s (acceptable for commands)
- small: ~3.5s (too slow for real-time)

## Complete Voice Control System

Putting it all together:

```python
class VoiceControlledRobot:
    def __init__(self):
        self.whisper = WhisperModel("base", device="cuda", compute_type="int8")
        self.wake_word_detector = WakeWordDetector("hey robot")
        self.command_executor = CommandExecutor()
        
    def run(self):
        """Main control loop"""
        print("Voice control system ready")
        
        while True:
            # Wait for wake word
            self.wake_word_detector.wait_for_wake_word()
            
            # Record command
            audio = self.record_command()
            
            # Transcribe
            text = self.transcribe(audio)
            print(f"Command: {text}")
            
            # Execute
            self.command_executor.execute(text)
    
    def transcribe(self, audio):
        segments, _ = self.whisper.transcribe(audio, language="en")
        return " ".join([s.text for s in segments])

# Run system
robot = VoiceControlledRobot()
robot.run()
```

## Summary

In this chapter, we covered:
- OpenAI Whisper fundamentals and model selection
- Real-time speech recognition implementation
- ROS 2 integration for robot control
- Advanced techniques (wake words, streaming, multilingual)
- Edge deployment on Jetson Orin Nano

**Next Chapter**: We'll integrate Whisper with LLMs for cognitive planning, enabling robots to understand complex, multi-step commands like "Clean the living room by organizing toys in the box and putting books on the shelf."