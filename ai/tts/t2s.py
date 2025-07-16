import os.path
import time
import threading
import sys

sys.path.append('../../')
from ai.tts.frontend.zh_frontend import Frontend
import onnxruntime as ort
import numpy as np
import simpleaudio as sa
import soundfile as sf
import torch


class Text2Speech():
    def __init__(self,
                 phones_dict="",
                 encoder_path="",
                 decoder_path="",
                 postnet_path="",
                 melgan_path="",
                 am_stat_path="",
                 gpu_id=0,
                 ):
        """
        TTS工具类参数
        :param phones_dict: 拼音文本路径
        :param encoder_path: 文本编码模型路径
        :param decoder_path: 文本解码模型路径
        :param postnet_path: postnet模型路径
        :param melgan_path: melgan模型路径
        :param am_stat_path: 归一化npz路径，存储了均值和方差
        :param gpu_id: <0为cpu,>=0为gpu
        """
        self.frontend = Frontend(
            phone_vocab_path=phones_dict,
            tone_vocab_path=None)

        if gpu_id >= 0:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            # 用CPU推理
            providers = ['CPUExecutionProvider']

        # 配置ort session
        sess_options = ort.SessionOptions()

        # 创建session
        self.am_encoder_infer_sess = ort.InferenceSession(encoder_path, providers=providers, sess_options=sess_options)

        ### 创建am_decoder_sess
        if 'onnx' in os.path.basename(decoder_path):
            self.decoder_backend = 'onnxruntime'
            self.am_decoder_sess = ort.InferenceSession(decoder_path, providers=providers, sess_options=sess_options)
        elif 'engine' in os.path.basename(decoder_path):
            import tensorrt as trt
            from ai.torch2trt import TRTModule
            self.decoder_backend = 'tensorrt'
            logger = trt.Logger(trt.Logger.INFO)
            with open(decoder_path, 'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            input_names = []
            output_names = []
            if trt.__version__.split('.')[0] == '10':
                # rt10.x
                for name in engine:
                    mode = engine.get_tensor_mode(name)
                    if mode == trt.TensorIOMode.INPUT:
                        input_names.append(name)
                    elif mode == trt.TensorIOMode.OUTPUT:
                        output_names.append(name)
            else:
                # rt8.x
                for i in range(engine.num_bindings):
                    if engine.binding_is_input(i):
                        input_names.append(engine.get_binding_name(i))
                    else:
                        output_names.append(engine.get_binding_name(i))
            self.am_decoder_sess = TRTModule(engine, input_names=input_names, output_names=output_names)
        else:
            self.decoder_backend = ''
            print('decoder backend error')

        ### 创建am_postnet_sess
        if 'onnx' in os.path.basename(postnet_path):
            self.postnet_backend = 'onnxruntime'
            self.am_postnet_sess = ort.InferenceSession(postnet_path, providers=providers, sess_options=sess_options)
        elif 'engine' in os.path.basename(postnet_path):
            import tensorrt as trt
            from ai.torch2trt import TRTModule
            self.postnet_backend = 'tensorrt'
            logger = trt.Logger(trt.Logger.INFO)
            with open(postnet_path, 'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())

            input_names = []
            output_names = []
            if trt.__version__.split('.')[0] == '10':
                # rt10.x
                for name in engine:
                    mode = engine.get_tensor_mode(name)
                    if mode == trt.TensorIOMode.INPUT:
                        input_names.append(name)
                    elif mode == trt.TensorIOMode.OUTPUT:
                        output_names.append(name)
            else:
                # rt8.x
                for i in range(engine.num_bindings):
                    if engine.binding_is_input(i):
                        input_names.append(engine.get_binding_name(i))
                    else:
                        output_names.append(engine.get_binding_name(i))
            self.am_postnet_sess = TRTModule(engine, input_names=input_names, output_names=output_names)
        else:
            self.postnet_backend = ''
            print('postnet backend error')

        ### 创建voc_melgan_sess
        if 'onnx' in os.path.basename(melgan_path):
            self.melgan_backend = 'onnxruntime'
            self.voc_melgan_sess = ort.InferenceSession(melgan_path, providers=providers, sess_options=sess_options)
        elif 'engine' in os.path.basename(melgan_path):
            import tensorrt as trt
            from ai.torch2trt import TRTModule
            self.melgan_backend = 'tensorrt'
            logger = trt.Logger(trt.Logger.INFO)
            with open(melgan_path, 'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())

            input_names = []
            output_names = []
            if trt.__version__.split('.')[0] == '10':
                # rt10.x
                for name in engine:
                    mode = engine.get_tensor_mode(name)
                    if mode == trt.TensorIOMode.INPUT:
                        input_names.append(name)
                    elif mode == trt.TensorIOMode.OUTPUT:
                        output_names.append(name)
            else:
                # rt8.x
                for i in range(engine.num_bindings):
                    if engine.binding_is_input(i):
                        input_names.append(engine.get_binding_name(i))
                    else:
                        output_names.append(engine.get_binding_name(i))
            self.voc_melgan_sess = TRTModule(engine, input_names=input_names, output_names=output_names)
        else:
            self.melgan_backend = ''
            print('melgan backend error')
        self.am_mu, self.am_std = np.load(am_stat_path)
        self.inference('你好啊世界')

        self._play_obj = None
        self._lock = threading.Lock()

    # 辅助函数 denorm, 训练过程中mel输出经过了norm，使用过程中需要进行denorm
    def denorm(self, data, mean, std):
        """stream am model need to denorm
        """
        return data * std + mean

    # 推理阶段封装
    # 端到端合成：一次性把句子全部合成完毕
    def inference(self, text):
        if text == "":
            print("text is empty")
            return None
        phone_ids = self.frontend.get_input_ids(text, merge_sentences=True, get_tone_ids=False)['phone_ids']
        t1 = time.time()

        orig_hs = self.am_encoder_infer_sess.run(None, input_feed={'text': phone_ids[0].numpy()})

        print('encoder', time.time() - t1)

        hs = orig_hs[0]
        t1 = time.time()
        if self.decoder_backend == 'onnxruntime':
            am_decoder_output = self.am_decoder_sess.run(None, input_feed={'xs': hs})
        else:
            input = torch.from_numpy(hs).to('cuda')
            with torch.no_grad():
                outputs = self.am_decoder_sess(input)
            am_decoder_output = [outputs.cpu().numpy()]
        print('decoder', time.time() - t1)
        t1 = time.time()
        xs = np.transpose(am_decoder_output[0], (0, 2, 1))
        if self.postnet_backend == 'onnxruntime':
            am_postnet_output = self.am_postnet_sess.run(None, input_feed={
                'xs': xs
            })
        else:
            input = torch.from_numpy(xs).to('cuda')
            with torch.no_grad():
                outputs = self.am_postnet_sess(input)
            am_postnet_output = [outputs.cpu().numpy()]

        print('postnet', time.time() - t1)

        am_output_data = am_decoder_output + np.transpose(am_postnet_output[0], (0, 2, 1))
        normalized_mel = am_output_data[0][0]
        mel = self.denorm(normalized_mel, self.am_mu, self.am_std)
        t1 = time.time()
        if self.melgan_backend == 'onnxruntime':
            wav = self.voc_melgan_sess.run(output_names=None, input_feed={'logmel': mel})[0]
        else:
            input = torch.from_numpy(mel).to('cuda')
            with torch.no_grad():
                outputs = self.voc_melgan_sess(input)
            wav = outputs.cpu().numpy()
        print('voc_melgan_sess', time.time() - t1)

        return wav

    def get_wav(self, wav, path="demo.wav", samplerate=24000, silence_sec=0.5):
        if wav.ndim == 1:
            silence = np.zeros(int(samplerate * silence_sec), dtype=wav.dtype)
        elif wav.ndim == 2:
            silence = np.zeros((int(samplerate * silence_sec), wav.shape[1]), dtype=wav.dtype)
        else:
            raise ValueError("Unsupported wav dimensions")
        wav_final = np.concatenate([wav, silence], axis=0)
        sf.write(path, wav_final, samplerate=24000)

    def threadingSpeak(self, wav, delayTime=0):
        threading.Timer(delayTime, self.speakText, args=(wav,)).start()

    def speakText(self, wav):
        try:
            audio_data_int16 = (wav * 32767).astype(np.int16)
            play_obj = sa.play_buffer(audio_data_int16, num_channels=1, bytes_per_sample=2, sample_rate=24000)
            with self._lock:
                self._play_obj = play_obj
            play_obj.wait_done()
            with self._lock:
                self._play_obj = None
        except Exception as e:
            print(f"An error occurred: {e}")

    def stop(self):
        with self._lock:
            if self._play_obj:
                self._play_obj.stop()
                self._play_obj = None


if __name__ == '__main__':
    phones_dict = r"D:\ai_library\ai\tts\fastspeech2_models\phone_id_map.txt"
    # 模型路径
    onnx_am_encoder = r"D:\ai_library\ai\tts\fastspeech2_models\fastspeech2_csmsc_am_encoder_infer.onnx"
    onnx_am_decoder = r"D:\ai_library\ai\tts\fastspeech2_models\fastspeech2_csmsc_am_decoder.onnx"
    onnx_am_postnet = r"D:\ai_library\ai\tts\fastspeech2_models\fastspeech2_csmsc_am_postnet.onnx"
    onnx_voc_melgan = r"D:\ai_library\ai\tts\melgan_models\mb_melgan_csmsc.onnx"

    am_stat_path = r"D:\ai_library\ai\tts\fastspeech2_models\speech_stats.npy"

    tts = Text2Speech(phones_dict, onnx_am_encoder, onnx_am_decoder, onnx_am_postnet, onnx_voc_melgan, am_stat_path)

    # infer_time = time.time()
    # wav = tts.inference('请投掷!')
    # print('infer time:', time.time() - infer_time)
    infer_time = time.time()
    wav = tts.inference('你的成绩是0分。')
    print('infer time:', time.time() - infer_time)
    # tts.get_wav(wav)

    tts.threadingSpeak(wav)
    time.sleep(1)
    tts.stop()