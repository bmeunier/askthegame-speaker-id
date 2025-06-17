# predict.py
import torch
from replicate.predictor import BasePredictor
from speechbrain.pretrained import SpeakerRecognition
from pydub import AudioSegment
import tempfile
import os

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and Alex's voiceprint into memory."""
        print("Loading speaker recognition model...")
        self.recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        print("Loading reference voiceprint...")
        self.alex_voiceprint = torch.load("alex_voiceprint.bin")
        print("Setup complete.")

    def predict(self, audio_url: str, start_time: float, end_time: float) -> str:
        """
        Takes an audio URL and a timestamp, extracts the clip,
        and compares its voiceprint to the reference.
        """
        print(f"Processing clip from {audio_url} at {start_time}s")
        # Download the full audio and clip it using pydub
        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_full_audio:
            os.system(f'curl -L -o {temp_full_audio.name} "{audio_url}"')
            
            sound = AudioSegment.from_mp3(temp_full_audio.name)
            # pydub works in milliseconds
            clip = sound[start_time * 1000 : end_time * 1000]
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_clip:
                clip.export(temp_clip.name, format="wav")
                
                # Create a voiceprint for the clip
                unknown_voiceprint = self.recognizer.encode_file(temp_clip.name)
        
        # Compare the voiceprints using cosine similarity
        similarity = torch.nn.CosineSimilarity(dim=-1)
        score = similarity(self.alex_voiceprint, unknown_voiceprint)
        
        print(f"Similarity score: {score.item()}")
        
        # Return "Alex Hormozi" if similarity is high, otherwise "Guest"
        if score.item() > 0.7: # We can tune this threshold later
            return "Alex Hormozi"
        else:
            return "Guest"