
from src.utils.init_path import init_path
from src.generate_facerender_batch import get_facerender_data
from src.generate_batch import get_data
from src.facerender.animate import AnimateFromCoeff
from src.test_audio2coeff import Audio2Coeff
from src.utils.preprocess import CropAndExtract
from pydub import AudioSegment
import torch
import uuid
import os
import shutil
import sys
sys.path.append('/data0/lizhong/SadTalker')

# from src.utils.preprocess import CropAndExtract
# from src.test_audio2coeff import Audio2Coeff
# from src.facerender.animate import AnimateFromCoeff
# from src.generate_batch import get_data
# from src.generate_facerender_batch import get_facerender_data
# from src.utils.init_path import init_path


# cur_dir = os.path.dirname(os.path.abspath(__file__))
# par_dir = os.path.dirname(cur_dir)
# os.chdir(par_dir)


def mp3_to_wav(mp3_filename, wav_filename, frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")


class SadTalker():

    def __init__(self, checkpoint_path='checkpoints', config_path='src/config', lazy_load=False):

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.device = device

        os.environ['TORCH_HOME'] = checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

    def test(self, source_image, driven_audio, preprocess='crop',
             still_mode=False,  use_enhancer=False, batch_size=1, size=256,
             pose_style=0, exp_scale=1.0,
             use_ref_video=False,
             ref_video=None,
             ref_info=None,
             use_idle_mode=False,
             length_of_audio=0, use_blink=True,
             #  result_dir='./results/'
             result_dir="./results"  # TODO
             ):

        self.sadtalker_paths = init_path(
            self.checkpoint_path,
            self.config_path,
            size, False,
            preprocess)
        # print(self.sadtalker_paths)

        # 预处理图像: 将人脸从图像中裁剪出来，并提取人脸的关键点以及3DMM形态
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        # 将音频数据转换为控制面部表情特别是唇部运动的系数
        self.preprocess_model = CropAndExtract(
            self.sadtalker_paths, self.device)
        # 根据上述两个模型数据生成最终的面部动画
        self.animate_from_coeff = AnimateFromCoeff(
            self.sadtalker_paths, self.device)

        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        print(source_image)
        pic_path = os.path.join(input_dir, os.path.basename(source_image))
        # shutil.move(source_image, input_dir)
        shutil.copy2(source_image, input_dir)  # TODO

        # WAV
        if driven_audio is not None and os.path.isfile(driven_audio):
            audio_path = os.path.join(
                input_dir, os.path.basename(driven_audio))

            # mp3 to wav
            if '.mp3' in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace(
                    '.mp3', '.wav'), 16000)
                audio_path = audio_path.replace('.mp3', '.wav')
            else:
                # shutil.move(driven_audio, input_dir)
                shutil.copy2(driven_audio, input_dir)  # TODO

        elif use_idle_mode:
            # generate audio from this new audio_path
            audio_path = os.path.join(
                input_dir, 'idlemode_'+str(length_of_audio)+'.wav')
            from pydub import AudioSegment
            one_sec_segment = AudioSegment.silent(
                duration=1000*length_of_audio)  # duration in milliseconds
            one_sec_segment.export(audio_path, format="wav")
        else:
            print(use_ref_video, ref_info)
            assert use_ref_video == True and ref_info == 'all'

        if use_ref_video and ref_info == 'all':  # full ref mode
            ref_video_videoname = os.path.basename(ref_video)
            audio_path = os.path.join(save_dir, ref_video_videoname+'.wav')
            print('new audiopath:', audio_path)
            # if ref_video contains audio, set the audio from ref_video.
            cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s" % (
                ref_video, audio_path)
            os.system(cmd)

        os.makedirs(save_dir, exist_ok=True)
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)

        # crop image and extract 3dmm from image
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
            pic_path,
            first_frame_dir,
            preprocess,
            True,
            size
        )

        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        if use_ref_video:
            print('using ref video for genreation')
            ref_video_videoname = os.path.splitext(
                os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_video_coeff_path, _, _ = self.preprocess_model.generate(
                ref_video, ref_video_frame_dir, preprocess, source_image_flag=False)
        else:
            ref_video_coeff_path = None

        if use_ref_video:
            if ref_info == 'pose':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = None
            elif ref_info == 'blink':
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'pose+blink':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'all':
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = None
            else:
                raise ('error in refinfo')
        else:
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None

        # audio2ceoff 将音频和其他输入数据转换为3DMM系数
        if use_ref_video and ref_info == 'all':
            # self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
            coeff_path = ref_video_coeff_path
        else:
            batch = get_data(first_coeff_path, audio_path, self.device,
                             ref_eyeblink_coeff_path=ref_eyeblink_coeff_path,
                             still=still_mode, idlemode=use_idle_mode,
                             length_of_audio=length_of_audio, use_blink=use_blink)  # longer audio?
            coeff_path = self.audio_to_coeff.generate(
                batch, save_dir, pose_style, ref_pose_coeff_path)

        # coeff2video  获取面部渲染数据
        data = get_facerender_data(
            coeff_path, crop_pic_path, first_coeff_path,
            audio_path, batch_size,
            still_mode=still_mode, preprocess=preprocess,
            size=size, expression_scale=exp_scale)

        print(f"\n>>>> Start create video....")
        # 生成动画视频
        return_path = self.animate_from_coeff.generate(
            data, save_dir,  pic_path, crop_info,
            enhancer='gfpgan' if use_enhancer else None,
            preprocess=preprocess, img_size=size)

        video_name = data['video_name']

        print(f'>>>>>The generated video is named {video_name} in {save_dir}')
        print(f">>> return_path is {return_path}")

        del self.preprocess_model
        del self.audio_to_coeff
        del self.animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import gc
        gc.collect()

        return return_path


if __name__ == "__main__":
    source_image = './examples/source_image/full_body_1.png'
    driven_audio = "audio_cache/audio_2024.05.23-21.47.58.wav"
    preprocess_type = "crop"
    is_still_mode = False
    enhancer = None
    batch_size = 2
    size_of_image = 256
    pose_style = 0

    checkpoint_path = "checkpoints"
    config_path = "src/config"
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)
    video_path = sad_talker.test(source_image, driven_audio, preprocess_type,
                                 is_still_mode, enhancer, batch_size, size_of_image,
                                 pose_style, result_dir="results")

    print(f"vedio_path is {video_path}")
    pass
