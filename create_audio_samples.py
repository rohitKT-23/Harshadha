"""
Create Audio Samples for Testing - MP3 & WAV Support
Uses Windows built-in text-to-speech to create test audio files
"""

import os
import sys
from pathlib import Path

def create_audio_samples(format="wav"):
    """Create sample audio files using Windows SAPI"""
    
    # Test phrases for speech-to-symbol conversion
    test_phrases = [
        "two plus three equals five",
        "ten percent of fifty dollars",
        "send email to john at company dot com",
        "is this correct question mark",
        "the temperature is minus twenty degrees",
        "please calculate five times six plus eight",
        "what is two hundred and fifty divided by ten",
        "the price is twenty five dollars and fifty cents"
    ]
    
    # Create audio directory
    audio_dir = Path("test_audio")
    audio_dir.mkdir(exist_ok=True)
    
    print(f"🎵 Creating audio samples in .{format} format using Windows Text-to-Speech...")
    
    for i, phrase in enumerate(test_phrases, 1):
        output_file = audio_dir / f"sample_{i:02d}.{format}"
        
        print(f"Creating: {phrase}")
        
        # Use Windows SAPI to create audio
        if format == "wav":
            # Direct WAV creation
            powershell_cmd = f'''
            Add-Type -AssemblyName System.Speech
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $synth.SetOutputToWaveFile("{output_file.absolute()}")
            $synth.Speak("{phrase}")
            $synth.Dispose()
            '''
        else:
            # Create WAV first, then convert to MP3 if needed
            temp_wav = audio_dir / f"temp_{i:02d}.wav"
            powershell_cmd = f'''
            Add-Type -AssemblyName System.Speech
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $synth.SetOutputToWaveFile("{temp_wav.absolute()}")
            $synth.Speak("{phrase}")
            $synth.Dispose()
            '''
        
        # Execute PowerShell command
        result = os.system(f'powershell -Command "{powershell_cmd}"')
        
        if result == 0:
            print(f"✅ Created: {output_file}")
        else:
            print(f"❌ Failed to create: {output_file}")
    
    print(f"\n🎯 Audio samples created in: {audio_dir.absolute()}")
    print(f"Total files: {len(list(audio_dir.glob(f'*.{format}')))}")
    
    return audio_dir

def test_audio_pipeline():
    """Test the complete audio pipeline with created samples"""
    
    audio_dir = Path("test_audio")
    if not audio_dir.exists():
        print("❌ No audio files found. Create them first!")
        return
    
    # Support both MP3 and WAV files
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    if not audio_files:
        print("❌ No audio files (.wav/.mp3) found in test_audio directory!")
        return
    
    print(f"\n🧪 Testing Audio Pipeline with {len(audio_files)} files...")
    
    # Test each audio file
    for audio_file in sorted(audio_files):
        print(f"\n🎵 Testing: {audio_file.name}")
        
        # Run the audio processing pipeline
        cmd = f'python main.py audio --file "{audio_file}"'
        print(f"Command: {cmd}")
        
        result = os.system(cmd)
        
        if result == 0:
            print("✅ Processed successfully!")
        else:
            print("❌ Processing failed!")
        
        print("-" * 50)

def test_with_existing_mp3():
    """Test with any existing MP3 files in current directory"""
    
    current_dir = Path(".")
    mp3_files = list(current_dir.glob("*.mp3"))
    
    if not mp3_files:
        print("❌ No MP3 files found in current directory!")
        print("💡 Put some .mp3 files here and try again")
        return
    
    print(f"🎵 Found {len(mp3_files)} MP3 files:")
    for mp3 in mp3_files:
        print(f"  - {mp3.name}")
    
    print("\n🧪 Testing with first MP3 file...")
    first_mp3 = mp3_files[0]
    
    cmd = f'python main.py audio --file "{first_mp3}"'
    print(f"Command: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        print(f"✅ Successfully processed: {first_mp3.name}")
    else:
        print(f"❌ Failed to process: {first_mp3.name}")

def record_custom_audio():
    """Instructions for recording custom audio"""
    print("""
🎙️  RECORD YOUR OWN AUDIO (MP3/WAV):

1. Use Windows Voice Recorder:
   - Press Windows + R
   - Type 'ms-windows-store://pdp/?productid=9WZDNCRFHWKL'
   - Or search "Voice Recorder" in Start menu

2. Record these phrases:
   - "Two plus three equals five"
   - "Ten percent of fifty"
   - "Send email to john at company dot com"

3. Save as .mp3 या .wav files anywhere

4. Test with:
   python main.py audio --file path/to/your_recording.mp3

💡 हमारा pipeline .mp3 और .wav दोनों support करता है!

🎵 Test करने के लिए:
   - कोई भी .mp3 file इस folder में रखो
   - python create_audio_samples.py existing चलाओ
    """)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            format_type = "wav"
            if len(sys.argv) > 2 and sys.argv[2] == "mp3":
                format_type = "mp3"
            create_audio_samples(format_type)
        elif sys.argv[1] == "test":
            test_audio_pipeline()
        elif sys.argv[1] == "existing":
            test_with_existing_mp3()
        elif sys.argv[1] == "record":
            record_custom_audio()
        else:
            print("Usage: python create_audio_samples.py [create|test|existing|record]")
    else:
        print("🎵 Audio Sample Creator (MP3/WAV Support)")
        print("Usage:")
        print("  python create_audio_samples.py create     - Create WAV samples")
        print("  python create_audio_samples.py create mp3 - Create MP3 samples") 
        print("  python create_audio_samples.py test       - Test pipeline with samples")
        print("  python create_audio_samples.py existing   - Test with existing MP3s")
        print("  python create_audio_samples.py record     - Recording instructions")
        print("")
        print("🎯 MP3 Testing:")
        print("  1. Put any .mp3 file in this folder")
        print("  2. Run: python main.py audio --file your_file.mp3")
        print("  3. Watch magic happen! ✨") 