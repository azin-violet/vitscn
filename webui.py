#coding:utf-8
import torch
import numpy as np
import argparse
import gradio as gr
import librosa

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from mel_processing import spectrogram_torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
lang = ['Chinese']

speaker_infos =  ['hutao',
                  'paimon',
                  'nahida', 
                  'zhongli', 
                  'yaeMiko',
                  'venti',
                  'klee']

speaker_to_id = {s: i for i, s in enumerate(speaker_infos)}
id_to_speaker = {i: s for i, s in enumerate(speaker_infos)}

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, required=True, help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True, help='Model path')
  args = parser.parse_args()

  hps = utils.get_hparams_from_file(args.config)

  net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)
  _ = net_g.eval()

  _ = utils.load_checkpoint(args.model, net_g, None)

  tts_fn = create_tts_fn(net_g, hps)
  vc_fn = create_vc_fn(net_g, hps)

  app = gr.Blocks()
  with app:
      with gr.Tab("Text-to-Speech"):
          with gr.Row():
              with gr.Column():
                  textbox = gr.TextArea(label="Text",
                                    #    lines=5,
                                       placeholder="Type your sentence here",
                                       value="原神, 启动!", elem_id=f"tts-input")
                  # select character
                  char_dropdown = gr.Dropdown(choices=speaker_infos, value=speaker_infos[0], label='character')
                  language_dropdown = gr.Dropdown(choices=lang, value=lang[0], label='language')
              with gr.Column():
                  text_output = gr.Textbox(label="Message")
                  audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                  btn = gr.Button("Generate")
                  btn.click(tts_fn,
                            inputs=[textbox, char_dropdown, language_dropdown],
                            outputs=[text_output, audio_output])
      with gr.Tab("Voice Conversion"):
          gr.Markdown("录制或上传声音，并选择要转换的音色。")
          with gr.Column():
              record_audio = gr.Audio(label="record your voice", source="microphone")
              upload_audio = gr.Audio(label="or upload audio here", source="upload")
              source_speaker = gr.Dropdown(choices=speaker_infos, value=speaker_infos[0], label="source speaker")
              target_speaker = gr.Dropdown(choices=speaker_infos, value=speaker_infos[0], label="target speaker")
          with gr.Column():
              message_box = gr.Textbox(label="Message")
              converted_audio = gr.Audio(label='converted audio')
          btn = gr.Button("Convert")
          btn.click(vc_fn, inputs=[source_speaker, target_speaker, record_audio, upload_audio], outputs=[message_box, converted_audio])
#   app.launch(share=False, server_name="0.0.0.0", server_port=7860)
  app.launch(share=False)

def create_tts_fn(model, hps):
    def tts_fn(text, speaker, language):
        if language is not None:
            pass  # to be added
        speaker_id = speaker_to_id[speaker]
        stn_tst = get_text(text, hps)
        with torch.no_grad():
          x_tst = stn_tst.to(device).unsqueeze(0)
          x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
          sid = torch.LongTensor([speaker_id]).to(device)
          audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)
    return tts_fn

def create_vc_fn(model, hps):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
      original_speaker_id = speaker_to_id[original_speaker]
      target_speaker_id = speaker_to_id[target_speaker]
      input_audio = record_audio if record_audio is not None else upload_audio
      if input_audio is None:
            return "You need to record or upload an audio", None
      sampling_rate, audio = input_audio
      original_speaker_id = speaker_to_id[original_speaker]
      target_speaker_id = speaker_to_id[target_speaker]
      if len(audio.shape) > 1:
          audio = librosa.to_mono(audio.astype('float').transpose(1, 0))
      if sampling_rate != hps.data.sampling_rate:
          audio = librosa.resample(audio.astype('float'), orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
      with torch.no_grad():
          y = torch.FloatTensor(audio)
          y = y / max(-y.min(), y.max()) / 0.99
          y = y.to(device)
          y = y.unsqueeze(0)
          spec = spectrogram_torch(y, hps.data.filter_length,
                                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                    center=False).to(device)
          spec_lengths = torch.LongTensor([spec.size(-1)]).to(device)
          sid_src = torch.LongTensor([original_speaker_id]).to(device)
          sid_tgt = torch.LongTensor([target_speaker_id]).to(device)
          audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][0, 0].data.cpu().float().numpy()
      del y, spec, spec_lengths, sid_src, sid_tgt
      return "Success", (hps.data.sampling_rate, audio)
    return vc_fn


if __name__ == '__main__':
  main()