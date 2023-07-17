# -*- coding: utf-8 -*-
from pathlib import Path
import datetime
import os
#TODO add aguments if needed
# import argparse
# parser = argparse.ArgumentParser(description='Audio to text script, using whisper or fast whisper')
# # Optional argument
# parser.add_argument('--dinput', type=int, help='A directory to read the audio files',default="./")
# parser.add_argument('--doutput', type=int, help='A directory to write the text files',default="./")

# create default input/output directories
input_dir='.\\audio'
# input_dir='C:\\Users\\lefra\\Downloads\\audio'
output_dir='.\\transcribed'
Path( output_dir ).mkdir( parents=True, exist_ok=True )
Path( input_dir ).mkdir( parents=True, exist_ok=True )


print("Ce script utilise faster whisper pour transcrire des fichiers audio.")
print("Utilisation: par défaut le script lit les fichiers dans le répertoire 'audio' situé là ou se trouve le script (il va le créer s'il n'existe pas)")
print("Par défaut le script écrit des fichiers txt des retranscriptions das le répertoire 'transcribed' situé là ou se trouve le script (il va le créer s'il n'existe pas)")
print("Pour changer les répertoires de lecture ou d'écriture: ouvrir le script et modifier la variable 'input_dir' pour les fichiers à lire et 'output_dir' pour le répertoire ou écreire les retranscriptions")
print("Vous pouvez utilsier des chemins relatifs ou absolus, exemples: ")
print("input_dir='.\\audio'")
print("input_dir='d:\\tests\\audio")
print("note: en chemin absolu, les '\\' doivent être doublés: '\\\\' ")

#check dependancies
try:
    from faster_whisper import WhisperModel
    import torch

except ImportError as err:
    print("Error loading modules, please check you installed all modules:")
    print("faster whisper: pip install faster-whisper")
    print("You also need to download ffmpeg and place it in this script directory : https://ffmpeg.org/download.html")
    print("Error catched: "+str(err))



def main():

    for audio_file in  os.listdir(path=input_dir):
        
        if str(audio_file).lower().endswith(".mp3") or str(audio_file).lower().endswith(".wav") or str(audio_file).lower().endswith(".m4a"):
            print ("Starting transcription of: "+str(audio_file))
            date_start = datetime.datetime.now()
            model_size = "large-v2"

            if torch.cuda.is_available():
                print("Cuda is available")
                _device = 'cuda'
                _compute_type="float16"
                # _compute_type="int8_float16"
            else:
                print("Cuda is not available, defaulting to using CPU")
                _device = "cpu"
                _compute_type="int8"
            
            model = WhisperModel(model_size, device=_device, compute_type=_compute_type)

            segments, info = model.transcribe(audio=input_dir+"/"+str(audio_file), beam_size=5)

            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

            _f = open(output_dir+"/"+str(audio_file)+".txt", "w",encoding="utf-8")
            _f.write("File:"+str(audio_file))
            _f.write(os.linesep)
            _f.write(f"Detected language: {info.language}")
            _f.write(os.linesep)
            _f.write(f"Transcribed text: ")

            for segment in segments:
                # this formula gives timestamps in seconds
                # f.write("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                # this formula translate timestamps in seconds to timestamps in hour min secs
                _f.write(f"["+ str(datetime.timedelta(seconds=segment.start))+"] -> [" + str(datetime.timedelta(seconds=segment.end))+"]"+ segment.text)
                _f.write(os.linesep)

            _f.close()
            date_end = datetime.datetime.now()
            tdelta = date_end - date_start 
            print (str(audio_file)+" has been transcribed in this timeframe: " +str(tdelta))
        else:
            print (str(audio_file)+ " is not a supported file, skipping it")
    


        

if __name__ == "__main__":
    main()

input("Appuyer une touche pour fermer le script. Les résultats se trouvent dans le réperoire: "+str(os.path.abspath(output_dir)))
