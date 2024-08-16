import zmq
import cv2
import modules.globals
import numpy as np
import threading
import time
import io
from tqdm import tqdm
from modules.typing import Face, Frame
from typing import Any,List
from modules.core import update_status
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
import zlib
import subprocess
from cv2 import VideoCapture
import queue
NAME = 'DLC.REMOTE-PROCESSOR'

context = zmq.Context()

# Socket to send messages on
def push_socket(address) -> zmq.Socket:
    sender_sock = context.socket(zmq.REQ)
    sender_sock.connect(address)
    return sender_sock
def pull_socket(address) -> zmq.Socket:
    sender_sock = context.socket(zmq.REP)
    sender_sock.connect(address)
    return sender_sock

def pre_check() -> bool:
    if not modules.globals.push_addr and not modules.globals.pull_addr:
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def stream_frame(temp_frame: Frame,stream_out: subprocess.Popen[bytes],stream_in: subprocess.Popen[bytes]) -> Frame:
    temp_framex = swap_face_remote(temp_frame,stream_out,stream_in)

    return temp_framex

def process_frame(source_frame: Frame, temp_frame: Frame)-> Frame:
    temp_framex = swap_frame_face_remote(source_frame,temp_frame)

    return temp_framex
def send_data(sender: zmq.Socket, face_bytes: bytes, metadata: dict, address: str) -> None:
    chunk_size = 1024*100
    total_chunk = len(face_bytes) // chunk_size + 1
    new_metadata = {'total_chunk': total_chunk}
    metadata.update(new_metadata)
    # Send metadata first
    sender.send_json(metadata)
    # Wait for acknowledgment for metadata
    ack = sender.recv_string()
    with tqdm(total=total_chunk, desc="Sending chunks", unit="chunk") as pbar:
        for i in range(total_chunk):
            chunk = face_bytes[i * chunk_size:(i + 1) * chunk_size]
            # Send the chunk
            sender.send(chunk)
            # Wait for acknowledgment after sending each chunk
            ack = sender.recv_string()
            pbar.set_postfix_str(f'Chunk {i + 1}/{total_chunk} ack: {ack}')
            pbar.update(1)
       
    # Send a final message to indicate all chunks are sent
    sender.send(b"END")
    # Wait for the final reply
    final_reply_message = sender.recv_string()
    print(f"Received final reply: {final_reply_message}")

def send_source_frame(source_face: Frame)-> None:
    sender = push_socket(modules.globals.push_addr)
    source_face_bytes = source_face.tobytes()
    metadata = {
            'manyface':(modules.globals.many_faces),
            'dtype_source':str(source_face.dtype),
            'shape_source':source_face.shape,
            'size':'640x480',
            'fps':'60'
            #'shape_temp':temp_frame.shape
        }
    send_data(sender, source_face_bytes, metadata,modules.globals.push_addr) 

def send_temp_frame(temp_face: Frame)-> None:
    sender = push_socket(modules.globals.push_addr_two)
    source_face_bytes = temp_face.tobytes()
    metadata = {
            'manyface':(modules.globals.many_faces),
            'dtype_temp':str(temp_face.dtype),
            'shape_temp':temp_face.shape,
            
            #'shape_temp':temp_frame.shape
        }
    send_data(sender, source_face_bytes, metadata,modules.globals.push_addr) 

def receive_processed_frame(output_queue: queue.Queue)-> None:
    while True:
        pull_socket_ = pull_socket(modules.globals.pull_addr)
        meta_data_json = pull_socket_.recv_json()
        print(meta_data_json)
        total_chunk = meta_data_json['total_chunk']
        # Send acknowledgment for metadata
        pull_socket_.send_string("ACK")
        # Receive the array bytes
        source_array_bytes =b'' 
        with tqdm(total=total_chunk, desc="Receiving chunks", unit="chunk") as pbar:
            for i in range(total_chunk):
                chunk = pull_socket_.recv()
                source_array_bytes += chunk
                pull_socket_.send_string(f"ACK {i + 1}/{total_chunk}")
                pbar.set_postfix_str(f'Chunk {i + 1}/{total_chunk}')
                pbar.update(1)
            

        end_message = pull_socket_.recv()
        if end_message == b"END":
            pull_socket_.send_string("Final ACK")
        
        # Deserialize the bytes back to an ndarray
        source_array = np.frombuffer(source_array_bytes, dtype=np.dtype(meta_data_json['dtype_source'])).reshape(meta_data_json['shape_source'])

        output_queue.put(source_array)  
        break
def send_streams(cap: VideoCapture) -> subprocess.Popen[bytes]:
    
    ffmpeg_command = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s',  f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        '-r',  str(int(cap.get(cv2.CAP_PROP_FPS))),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-rtbufsize', '100M',
        '-f', 'mpegts', modules.globals.push_addr_two #'tcp://127.0.0.1:5552'
    ]


    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    return ffmpeg_process
def recieve_streams(cap: VideoCapture)->subprocess.Popen[bytes]:
    ffmpeg_command_recie = [
    'ffmpeg',
    '-i',modules.globals.pull_addr, #'tcp://127.0.0.1:5553',
    '-f','rawvideo',
    '-pix_fmt','bgr24',
    '-s','960x540',#'640x480',
    'pipe:1'
    ]
    
    ffmpeg_process_com = subprocess.Popen(ffmpeg_command_recie, stdout=subprocess.PIPE)
    return ffmpeg_process_com

def write_to_stdin(queue: queue.Queue, stream_out: subprocess.Popen):
   
    temp_frame = queue.get()
    temp_frame_bytes = temp_frame.tobytes()
    stream_out.stdin.write(temp_frame_bytes)
def read_from_stdout(queue: queue.Queue, stream_in: subprocess.Popen, output_queue: queue.Queue):
    
    raw_frame = stream_in.stdout.read(960 * 540 * 3)
    
    
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((540, 960, 3))
    output_queue.put(frame)  
def swap_face_remote(temp_frame: Frame,stream_out:subprocess.Popen[bytes],stream_in: subprocess.Popen[bytes]) -> Frame:
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    # Start threads for stdin and stdout
    write_thread = threading.Thread(target=write_to_stdin, args=(input_queue, stream_out))
    read_thread = threading.Thread(target=read_from_stdout, args=(input_queue, stream_in, output_queue))
    
    write_thread.start()
    read_thread.start()
    
    # Send the frame to the stdin thread
    input_queue.put(temp_frame)
    
    # Wait for the processed frame from the stdout thread
    processed_frame = output_queue.get()
    
    # Stop the threads
    input_queue.put(None)
    write_thread.join()
    read_thread.join()

    return processed_frame
    
    
def swap_frame_face_remote(source_frame: Frame,temp_frame: Frame) -> Frame:
    #input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    # Start threads for stdin and stdout
    write_thread = threading.Thread(target=send_source_frame, args=(source_frame,))
    write_thread_tw = threading.Thread(target=send_temp_frame, args=(temp_frame,))
    read_thread_ = threading.Thread(target=receive_processed_frame, args=(output_queue,))
    
    write_thread.start()
    write_thread_tw.start()
    read_thread_.start()
    
    # Send the frame to the stdin thread
    
    # Wait for the processed frame from the stdout thread
    processed_frame = output_queue.get()
    
    # Stop the threads
    write_thread.join()
    write_thread_tw.join()
    read_thread_.join()

    return processed_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
