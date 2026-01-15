# multimedia_processor.py
import os
import tempfile
from typing import Tuple, Dict, Any, Optional
import streamlit as st

# Video/Audio processing
try:
    import moviepy.editor as mp
except ImportError:
    mp = None
    st.warning("moviepy not installed. Video processing will be limited.")

# Audio processing
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
except ImportError:
    sr = None
    AudioSegment = None
    split_on_silence = None
    st.warning("speech_recognition/pydub not installed. Audio processing will be limited.")

# Image processing
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    pytesseract = None
    Image = None
    st.warning("pytesseract/PIL not installed. Image OCR will be limited.")

class MultimediaProcessor:
    """Process multimedia files (video, audio, images) to extract text"""

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        """Check and warn about missing dependencies"""
        missing = []

        if mp is None:
            missing.append("moviepy (for video processing)")
        if sr is None or AudioSegment is None:
            missing.append("speech-recognition/pydub (for audio processing)")
        if pytesseract is None or Image is None:
            missing.append("pytesseract/Pillow (for image OCR)")

        if missing:
            st.warning(f"Missing dependencies: {', '.join(missing)}")
            st.info("Install with: pip install moviepy speechrecognition pydub pytesseract pillow")

    @staticmethod
    def extract_audio_from_video(video_path: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extract audio from video file using moviepy"""
        debug_info = {
            "method": "moviepy",
            "success": False,
            "duration": 0,
            "audio_format": "wav"
        }

        if mp is None:
            debug_info["error"] = "moviepy not installed"
            st.error("moviepy is required for video processing. Install with: pip install moviepy")
            return None, debug_info

        try:
            # Load video
            with st.spinner("Loading video..."):
                video = mp.VideoFileClip(video_path)
                debug_info["duration"] = video.duration

                if video.audio is None:
                    debug_info["error"] = "No audio track found in video"
                    st.warning("No audio found in video file")
                    video.close()
                    return None, debug_info

            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                audio_path = tmp_audio.name

            # Extract and save audio
            with st.spinner("Extracting audio..."):
                video.audio.write_audiofile(
                    audio_path,
                    codec='pcm_s16le',  # Standard WAV format
                    fps=44100,  # CD quality
                    logger=None  # Suppress moviepy logs
                )
                video.close()

            debug_info["success"] = True
            debug_info["audio_file"] = audio_path
            debug_info["audio_size_mb"] = os.path.getsize(audio_path) / (1024 * 1024)

            st.success(f"✓ Audio extracted: {debug_info['duration']:.1f}s, {debug_info['audio_size_mb']:.2f}MB")
            return audio_path, debug_info

        except Exception as e:
            debug_info["error"] = str(e)
            st.error(f"Audio extraction failed: {e}")
            return None, debug_info

    @staticmethod
    def transcribe_audio(audio_path: str, language: str = "en-US") -> Tuple[str, Dict[str, Any]]:
        """Transcribe audio to text using speech recognition"""
        debug_info = {
            "method": "speech_recognition",
            "success": False,
            "language": language,
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_chunks": 0
        }

        if sr is None or AudioSegment is None:
            debug_info["error"] = "speech_recognition/pydub not installed"
            st.error("Required packages not installed. Install with: pip install speechrecognition pydub")
            return "", debug_info

        try:
            recognizer = sr.Recognizer()

            # Load audio file
            with st.spinner("Loading audio file..."):
                audio = AudioSegment.from_file(audio_path)
                debug_info["duration_seconds"] = len(audio) / 1000
                debug_info["sample_rate"] = audio.frame_rate

            # Split audio on silence for better accuracy
            with st.spinner("Splitting audio into chunks..."):
                chunks = split_on_silence(
                    audio,
                    min_silence_len=500,  # Minimum silence length in ms
                    silence_thresh=audio.dBFS - 14,  # Silence threshold
                    keep_silence=250  # Keep some silence at boundaries
                )

                # If no chunks from silence splitting, split by duration
                if not chunks:
                    chunk_duration = 30000  # 30 seconds
                    chunks = [audio[i:i + chunk_duration]
                            for i in range(0, len(audio), chunk_duration)]

            debug_info["total_chunks"] = len(chunks)
            transcript_parts = []

            # Process each chunk
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, chunk in enumerate(chunks):
                chunk_num = i + 1
                status_text.text(f"Transcribing chunk {chunk_num}/{len(chunks)}...")

                # Export chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_chunk:
                    chunk_path = tmp_chunk.name

                try:
                    # Export chunk
                    chunk.export(chunk_path, format="wav")

                    # Transcribe chunk
                    with sr.AudioFile(chunk_path) as source:
                        # Adjust for ambient noise
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)

                        # Read audio data
                        audio_data = recognizer.record(source)

                        # Try Google Speech Recognition
                        try:
                            text = recognizer.recognize_google(
                                audio_data,
                                language=language
                            )
                            transcript_parts.append(text)
                            debug_info["successful_chunks"] += 1

                        except sr.UnknownValueError:
                            debug_info["failed_chunks"] += 1
                            st.warning(f"Chunk {chunk_num}: Speech not understood")

                        except sr.RequestError as e:
                            debug_info["failed_chunks"] += 1
                            st.warning(f"Chunk {chunk_num}: API error - {e}")

                finally:
                    # Clean up temp file
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)

                # Update progress
                progress_bar.progress((i + 1) / len(chunks))

            progress_bar.empty()
            status_text.empty()

            # Combine all transcript parts
            transcript = " ".join(transcript_parts)
            debug_info["transcript_length"] = len(transcript)
            debug_info["transcript_words"] = len(transcript.split())

            if transcript:
                debug_info["success"] = True
                success_rate = (debug_info["successful_chunks"] / debug_info["total_chunks"]) * 100
                st.success(f"✓ Transcription complete: {debug_info['transcript_words']} words ({success_rate:.1f}% success rate)")
            else:
                debug_info["error"] = "No text transcribed"
                st.warning("No text could be transcribed from audio")

            return transcript.strip(), debug_info

        except Exception as e:
            debug_info["error"] = str(e)
            st.error(f"Transcription failed: {e}")
            return "", debug_info

    @staticmethod
    def process_video_to_text(video_path: str) -> Tuple[str, Dict[str, Any]]:
        """Complete pipeline: video → audio → text"""
        debug_info = {
            "pipeline": "video → audio → text",
            "stages": []
        }

        # Stage 1: Extract audio
        st.info("🔊 Extracting audio from video...")
        audio_path, audio_debug = MultimediaProcessor.extract_audio_from_video(video_path)
        debug_info["stages"].append({"stage": "audio_extraction", **audio_debug})

        if not audio_path or not audio_debug.get("success"):
            debug_info["error"] = "Audio extraction failed"
            return "", debug_info

        # Stage 2: Transcribe audio
        st.info("📝 Transcribing audio to text...")
        transcript, transcription_debug = MultimediaProcessor.transcribe_audio(audio_path)
        debug_info["stages"].append({"stage": "transcription", **transcription_debug})

        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        if transcript:
            debug_info["success"] = True
            debug_info["final_text_length"] = len(transcript)
            debug_info["final_word_count"] = len(transcript.split())
        else:
            debug_info["error"] = "Transcription failed"

        return transcript, debug_info

    @staticmethod
    def extract_text_from_image(image_path: str, preprocess: bool = True,
                            language: str = "eng") -> Tuple[str, Dict[str, Any]]:
    """Extract text from image using OCR with preprocessing"""
    debug_info = {
        "method": "pytesseract_ocr",
        "preprocessing": preprocess,
        "language": language,
        "success": False
    }

    if pytesseract is None or Image is None:
        debug_info["error"] = "pytesseract/Pillow not installed"
        st.error("Required packages not installed. Install with: pip install pytesseract pillow")
        return "", debug_info

    try:
        # Load image
        with st.spinner("Loading image..."):
            image = Image.open(image_path)
            debug_info["original_size"] = image.size
            debug_info["original_mode"] = image.mode

        # Optional preprocessing for better OCR
        if preprocess:
            with st.spinner("Preprocessing image for better OCR..."):
                # Convert to grayscale if not already
                if image.mode != 'L':
                    image = image.convert('L')

                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)  # Increase contrast

                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2.0)  # Increase sharpness

                # Optional: Apply slight blur to reduce noise
                image = image.filter(ImageFilter.MedianFilter(size=3))

                debug_info["preprocessing_applied"] = True

        # Configure OCR for better accuracy
        custom_config = r'--oem 3 --psm 6'

        # Extract text
        with st.spinner("Running OCR..."):
            text = pytesseract.image_to_string(
                image,
                config=custom_config,
                lang=language
            )

        debug_info["text_length"] = len(text.strip())
        debug_info["word_count"] = len(text.strip().split())

        if text.strip():
            debug_info["success"] = True
            st.success(f"✓ OCR complete: {debug_info['word_count']} words extracted")

            # Show OCR confidence if available
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [float(c) for c in data['conf'] if float(c) > 0]
                if confidences:
                    debug_info["avg_confidence"] = sum(confidences) / len(confidences)
                    st.info(f"Average OCR confidence: {debug_info['avg_confidence']:.1f}%")
            except:
                pass
        else:
            debug_info["warning"] = "No text found in image"
            st.warning("No text could be extracted from image")

        return text.strip(), debug_info

    except Exception as e:
        debug_info["error"] = str(e)
        st.error(f"Image OCR failed: {e}")
        return "", debug_info

@staticmethod
def process_audio_to_text(audio_path: str, language: str = "en-US") -> Tuple[str, Dict[str, Any]]:
    """Process audio file to text"""
    return MultimediaProcessor.transcribe_audio(audio_path, language)

@classmethod
def process_file(cls, file_path: str, filename: str) -> Tuple[str, Dict[str, Any]]:
    """Process multimedia file based on extension"""
    file_ext = filename.lower().split('.')[-1]

    if file_ext in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']:
        return cls.process_video_to_text(file_path)

    elif file_ext in ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac']:
        return cls.process_audio_to_text(file_path)

    elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif']:
        return cls.extract_text_from_image(file_path)

    else:
        raise ValueError(f"Unsupported multimedia format: {file_ext}")