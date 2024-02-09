
from transformers import AutoProcessor, MusicgenMelodyForCausalLM, MusicgenMelodyForConditionalGeneration

processor = AutoProcessor.from_pretrained("ylacombe/musicgen-melody-bella-ciao")
model = MusicgenMelodyForCausalLM.from_pretrained("ylacombe/musicgen-melody-bella-ciao")

bigger_model = MusicgenMelodyForConditionalGeneration.from_pretrained("ylacombe/musicgen-melody")

bigger_model.decoder = model
bigger_model.config.decoder = model.config

inputs = processor(
    text=["80s blues track with groovy saxophone"],
    padding=True,
    return_tensors="pt",
)
audio_values = bigger_model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

import soundfile as sf

sampling_rate = bigger_model.config.audio_encoder.sampling_rate
sf.write("musicgen_out.wav", audio_values[0].T.numpy(), sampling_rate)
