# mpdi - mp3 to MIDI

A sophisticated Python application that converts vocal melodies from audio files into MIDI format using multiple advanced pitch detection algorithms. The name "MPDI" stands for **M**usical **P**itch **D**etection **I**nterface.


## Features

### 🎵 Advanced Pitch Detection
- **Multiple Algorithms**: Choose between CREPE (neural network), PYIN, or combined methods
- **High Accuracy**: State-of-the-art machine learning models for precise note detection
- **Noise Reduction**: Built-in audio cleaning and noise suppression
- **Harmonic Analysis**: Isolates vocal components from background sounds

### 🎹 Intelligent MIDI Conversion
- **Note Smoothing**: Eliminates pitch wobble and merges short identical notes
- **Duration Filtering**: Removes artifacts and maintains musical phrasing
- **Velocity Control**: Dynamic note velocity based on audio loudness
- **Instrument Selection**: Export to various MIDI instruments

### 🖥️ User-Friendly Interface
- **Modern GUI**: Built with Tkinter for cross-platform compatibility
- **Real-time Progress**: Visual feedback during conversion process
- **Adjustable Parameters**: Fine-tune sensitivity, thresholds, and ranges
- **Batch Processing**: Support for multiple audio formats

## Supported Input Formats
- MP3
- WAV
- OGG
- M4A
- FLAC

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mpdi.git
   cd mpdi
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Manual Installation
If you prefer to install packages manually:

```bash
pip install librosa numpy pretty_midi scipy noisereduce crepe-python tkinter
```

## Usage

### Basic Usage
1. Launch the application:
   ```bash
   python mpdi.py
   ```

2. Select your input audio file
3. Choose output MIDI file location
4. Adjust settings as needed
5. Click "Convert" and wait for processing

### Algorithm Selection

| Method | Best For | Accuracy | Speed |
|--------|----------|----------|-------|
| **CREPE** | Complex recordings, noisy environments | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **PYIN** | Clean vocals, fast processing | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Combined** | Balanced approach, reliable results | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### Recommended Settings

**For Male Voices:**
- Range: C2 - C5
- Sensitivity: 0.6-0.8
- Min Duration: 0.1s

**For Female Voices:**
- Range: A3 - C6  
- Sensitivity: 0.7-0.9
- Min Duration: 0.08s

**For Noisy Recordings:**
- Enable Noise Reduction
- Increase Volume Threshold (0.03-0.05)
- Use CREPE or Combined method

## Technical Details

### Algorithms Used

#### CREPE (Convolutional REpresentation for Pitch Estimation)
- Deep learning-based pitch detection
- Trained on large dataset of musical audio
- Excellent for complex vocal performances
- Requires more computational resources

#### PYIN (Probabilistic YIN)
- Probabilistic extension of classic YIN algorithm
- Robust to noise and artifacts
- Faster processing time
- Ideal for clean vocal recordings

#### Advanced Post-Processing
- Median filtering for outlier removal
- Gaussian smoothing for natural transitions
- RMS-based volume filtering
- Note quantization and duration optimization

### Performance Notes
- Processing time varies with audio length and algorithm choice
- CREPE may take 2-3x longer than PYIN
- 1 minute of audio typically processes in 1-5 minutes
- RAM usage: 500MB - 2GB depending on file size

## Troubleshooting

### Common Issues

**"No notes detected in audio"**
- Check if vocal range matches the singer's voice
- Increase sensitivity setting
- Enable noise reduction for noisy recordings
- Verify audio file isn't corrupted

**"CREPE algorithm failed"**
- Ensure crepe-python is properly installed
- Try PYIN or combined method as fallback
- Check available RAM (CREPE requires more memory)

**"Poor note accuracy"**
- Adjust the note range to match the vocalist
- Increase minimum note duration to 0.1s
- Enable harmonic/percussive separation
- Use combined method for better results

**"Application crashes with large files"**
- Close other memory-intensive applications
- Split long audio files into shorter segments
- Use PYIN algorithm for faster processing

### System Requirements
- **Minimum**: 4GB RAM, 2GHz dual-core processor
- **Recommended**: 8GB RAM, 3GHz quad-core processor
- **Storage**: 1GB free space for temporary files

## Output

MPDI generates standard MIDI files (.mid) that can be:
- Imported into DAWs (Ableton, FL Studio, Logic Pro, etc.)
- Edited in notation software (MuseScore, Sibelius, Finale)
- Used with virtual instruments and synthesizers
- Further processed in music production workflows

## Project Structure

```
mpdi/
├── mpdi.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── examples/              # Example files
    ├── sample_vocal.mp3   # Example input
    └── sample_output.mid  # Example output
```

## Contributing

We welcome contributions to MPDI! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Librosa** team for audio analysis tools
- **CREPE** developers for the pitch detection model
- **PrettyMIDI** for MIDI file handling
- The open-source music technology community

## Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**MPDI** - Transforming vocal performances into musical notation through advanced computational analysis.
