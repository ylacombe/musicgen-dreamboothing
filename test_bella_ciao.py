
from transformers import AutoProcessor, MusicgenMelodyForCausalLM, MusicgenMelodyForConditionalGeneration, set_seed

processor = AutoProcessor.from_pretrained("ylacombe/musicgen-melody") #ylacombe/musicgen-melody-bella-ciao")
# model = MusicgenMelodyForCausalLM.from_pretrained("ylacombe/musicgen-melody") #ylacombe/musicgen-melody-bella-ciao")

bigger_model = MusicgenMelodyForConditionalGeneration.from_pretrained("ylacombe/musicgen-melody").to("cuda:3")

# bigger_model.decoder = model.to("cuda:3")
# bigger_model.config.decoder = model.config

inputs = processor(
    text=["bella ciao 80s blues"], # track with groovy saxophone"],
    padding=True,
    return_tensors="pt",
).to("cuda:3")

set_seed(0)
audio_values = bigger_model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=512)

import soundfile as sf

sampling_rate = bigger_model.config.audio_encoder.sampling_rate
sf.write("musicgen_out.wav", audio_values[0].T.cpu().numpy(), sampling_rate)
