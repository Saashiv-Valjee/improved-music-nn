import os
import shutil
import time
from os import listdir
from pathlib import Path
from tempfile import NamedTemporaryFile
import pandas as pd

import tensorflow as tf
from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from magenta.music import midi_io, music_pb2
from note_seq import midi_io as Nmid
from note_seq.protobuf import generator_pb2
from note_seq import sequences_lib
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles

from Models.MyLSTM import Gen_LSTM, mse_with_positive_pressure
from Music_Processor.MusicProcessor import MusicProcessor

# uvicorn Midi-Generator.app:MyApp --reload

MyApp = FastAPI()

# Create a Jinja2Templates instance
templates = Jinja2Templates(directory="templates")

processor = MusicProcessor()

tf.keras.utils.get_custom_objects()['mse_with_positive_pressure'] = mse_with_positive_pressure

model_path = r'Data/path_to_save_model'
model = tf.keras.models.load_model(model_path)

MyApp.mount("/images", StaticFiles(directory="images"), name="images")
MyApp.mount("/midis", StaticFiles(directory="midis"), name="midis")

@MyApp.get("/list_midis/")
async def list_midis():
    midi_dir = Path("midis")
    midi_files = [f.name for f in midi_dir.iterdir() if f.suffix == '.midi']
    return {"midi_files": midi_files}

def delete_file_later(file_path: Path, delay: int = 60):
    print(f'deleting file {file_path} in 60 seconds')
    time.sleep(delay)  # wait for the given delay (in seconds)
    try:
        if file_path.exists():
            file_path.unlink()
            print(f"File {file_path} has been deleted.")
    except Exception as e:
        print(f"An error occurred while deleting {file_path}: {e}")

@MyApp.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@MyApp.get("/download/{filename}")
async def download_midi(filename: str, background_tasks: BackgroundTasks):
    file_path = Path(f"Midi-Generator/output/{filename}-generated.midi")
    print(f"Attempting to fetch file at: {file_path}")
    if file_path.exists():
        file_like = file_path.open("rb")
        response = StreamingResponse(file_like, media_type="audio/midi")

        # Schedule the deletion of the file after a delay
        background_tasks.add_task(delete_file_later, file_path, delay=60)

        return response
    else:
        return {"error": "File not found"}


@MyApp.post("/upload_midi/")
async def upload_midi(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...),):
    try:
        # Create a temporary file to store the uploaded MIDI file
        temp_file = NamedTemporaryFile(delete=False)
        with temp_file as buffer:
            # Copy the uploaded file to the temporary file
            shutil.copyfileobj(file.file, buffer)

        sample_file = temp_file.name
        filename_without_extension = Path(file.filename).stem
        print(f"Processing {sample_file}")
        instrument_name = processor.get_name(midi_file=sample_file)
        sample_notes = processor.preprocess(sample_file)

        all_notes_df = processor.midi_to_notes(sample_file)
        end_time_of_last_note = all_notes_df['end'].max()
        start_time_threshold = end_time_of_last_note - 10
        last_seconds_df = all_notes_df[all_notes_df['start'] >= start_time_threshold]
        offset_value = last_seconds_df['start'].min()
        last_seconds_df['start'] = last_seconds_df['start'] - offset_value
        last_seconds_df['end'] = last_seconds_df['end'] - offset_value
        start_offset_for_generated = last_seconds_df['end'].max() + 1

        generated_notes = Gen_LSTM(model, sample_notes)

        generated_notes['start'] = generated_notes['start'] + start_offset_for_generated
        generated_notes['end'] = generated_notes['end'] + start_offset_for_generated

        combined_notes = pd.concat([last_seconds_df, generated_notes], ignore_index=True)

        # Calculate the interval from 0 to the first 'start' value
        interval = combined_notes['start'].iloc[0]
        print(interval)

        # Subtract this interval from 'start' and 'end' columns
        combined_notes['start'] -= interval
        combined_notes['end'] -= interval

        interval = combined_notes['start'].iloc[0]
        print(interval)

        output_dir = Path(r"Midi-Generator/output")
        output_dir.mkdir(exist_ok=True)
        out_file_path = output_dir / f"{filename_without_extension}-generated.midi"

        # Convert generated notes to MIDI and save it
        processor.notes_to_midi(combined_notes, instrument_name=instrument_name, out_file=str(out_file_path))

        print(f"Generated file saved at: {out_file_path}")
        print(f"Filename without extension: {filename_without_extension}")

        os.remove(sample_file)
        background_tasks.add_task(delete_file_later, out_file_path, delay=60)
        return templates.TemplateResponse("success.html", {"request": request, "filename_without_extension": filename_without_extension})

    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": str(e)}
    
@MyApp.post("/upload_midi_magenta/")
async def upload_midi_magenta(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded MIDI file
        temp_file = NamedTemporaryFile(delete=False)
        with temp_file as buffer:
            # Copy the uploaded file to the temporary file
            shutil.copyfileobj(file.file, buffer)

        sample_file = temp_file.name
        filename_without_extension = Path(file.filename).stem
        print(f"Processing {sample_file}")

        # Preprocess the MIDI file using Magenta
        primer_seq = Nmid.midi_file_to_note_sequence(sample_file)
        primer_duration = 10 
        primer = MusicProcessor.trim_to_last_n_seconds(primer_seq,primer_duration)

        # Define the path to your .mag file
        bundle_file = 'Models/performance_with_dynamics.mag'

        # Load the model
        bundle = sequence_generator_bundle.read_bundle_file(bundle_file)
        generator_map = performance_sequence_generator.get_generator_map()
        performance_rnn = generator_map['performance_with_dynamics'](checkpoint=None, bundle=bundle)
        performance_rnn.initialize()

        # Define temperature and qpm
        temperature = 1.0  # Controls the randomness. Higher values make the output more random.
        qpm = 150  # Speed of the generated sequence

        generator_options = generator_pb2.GeneratorOptions()
        generate_section = generator_options.generate_sections.add()
        generate_section.start_time = primer_seq.total_time
        generate_section.end_time = primer_seq.total_time + 30  #adjust this to determine the length of the generated sequence
        generator_options.args['temperature'].float_value = temperature
        generator_options.args['qpm'].float_value = qpm

        # Generate the sequence
        generated_sequence = performance_rnn.generate(primer, generator_options)
        Gfinal_sequence = MusicProcessor.generation_offset(generated_sequence,primer_seq.total_time)
        combined_sequence = sequences_lib.concatenate_sequences([primer, Gfinal_sequence])

        output_dir = Path(r"Midi-Generator/output")
        output_dir.mkdir(exist_ok=True)
        out_file_path = output_dir / f"{filename_without_extension}-generated.midi"

        midi_io.note_sequence_to_midi_file(combined_sequence, out_file_path)

        # Now it's safe to remove the temporary copy of the uploaded MIDI file
        os.remove(sample_file)
        background_tasks.add_task(delete_file_later, out_file_path, delay=60)
        return templates.TemplateResponse("success.html", {"request": request, "filename_without_extension": filename_without_extension})
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": str(e)}

@MyApp.get('/Midi-Generator/output/{filename}')
async def serve_midi(filename: str):
    file_path = Path(f"Midi-Generator/output/{filename}")
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path, media_type="audio/midi")
    else:
        raise HTTPException(status_code=404, detail="File not found")

