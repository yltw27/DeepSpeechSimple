# DeepSpeech_simple

Clone from mozilla's deepspeech project: https://github.com/mozilla/DeepSpeech
You could train your own "simple commands recognition" with ~500 .wav files.

---------------------------------------------------------------------------------------
Steps:
1. Download https://github.com/mozilla/DeepSpeech

2. Edit data/self/recording_scripts.txt and run recording_deepspeech.py to collect your data
   (or you can download open source "LibriVox" from http://www.openslr.org/12/ 
    and run flac2wav.py )
    
3. Edit and run catalog.py to create a csv file for training

4. Edit and run DeepSpeecg.py with following settings:
    train_files/ dev_file/ test_files: path to the csv file you create with catalog.py
    checkpoint_dir: where you want to save the model 
    max_to_keep: 1
    n_hidden: 128 (or you can use 256, 512... larger hidden size will cause longer inference time)
    
    *you can also insert this block to line 10 
     and use " 'ckpt/'+ckpt_folder " as the checkpoint_dir:

    start = datatime.datetime.now()
    ckpt_folder = start.strftime("%Y-%m-%d-%H%M")
    try:
        os.mkdir('ckpt/'+ckpt_folder)
    except OSError:
        pass

5. Edit and run inference.py with the checkpoint folder(model) you trained  
