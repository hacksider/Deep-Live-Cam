import os
import cv2
import shutil
import time
import base64
import json # For parsing target_ids_json
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import modules.globals as Globals
import modules.core as core
from modules.utilities import normalize_output_path, get_temp_directory_path, is_image as util_is_image
from modules.face_analyser import get_one_face, get_many_faces, get_unique_faces_from_target_image, simplify_maps # Added simplify_maps
import modules.processors.frame.core as frame_processors_core

VIDEO_CAMERA = None
target_path_web = None
prev_time = 0
frame_count = 0 # For FPS calculation
current_fps = 0 # For FPS calculation

# Attempt to load initial settings from a file if it exists
# This is a placeholder for more sophisticated settings management.
# For now, we rely on defaults in modules.globals or explicit setting via UI.
# if os.path.exists('switch_states.json'):
#     try:
#         with open('switch_states.json', 'r') as f:
#             import json
#             states = json.load(f)
#             # Assuming states directly map to Globals attributes
#             for key, value in states.items():
#                 if hasattr(Globals, key):
#                     setattr(Globals, key, value)
#     except Exception as e:
#         print(f"Error loading switch_states.json: {e}")


app = Flask(__name__)
CORS(app) # Enable CORS for all routes

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
PROCESSED_OUTPUTS_FOLDER = os.path.join(os.getcwd(), 'processed_outputs') # Added
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_OUTPUTS_FOLDER): # Added
    os.makedirs(PROCESSED_OUTPUTS_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_OUTPUTS_FOLDER'] = PROCESSED_OUTPUTS_FOLDER # Added

@app.route('/')
def index(): # Renamed from hello_world
    return render_template('index.html')

@app.route('/upload/source', methods=['POST'])
def upload_source():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        Globals.source_path = filepath
        return jsonify({'message': 'Source uploaded', 'filepath': filepath}), 200

@app.route('/upload/target', methods=['POST'])
def upload_target():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    global target_path_web # Use the web-specific target path
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        Globals.target_path = filepath # This is for the core processing engine
        target_path_web = filepath # This is for UI state, like triggering face mapping
        # Provide a URL to the uploaded file for preview if desired, requires a new endpoint or serving 'uploads' statically
        # For now, client-side preview is used.
        return jsonify({'message': 'Target uploaded', 'filepath': filepath, 'file_url': f'/uploads/{filename}'}), 200


@app.route('/uploads/<filename>') # Simple endpoint to serve uploaded files for preview
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Update Globals based on received data
    # Example:
    if 'keep_fps' in data:
        Globals.keep_fps = bool(data['keep_fps'])
    if 'keep_audio' in data:
        Globals.keep_audio = bool(data['keep_audio'])
    if 'many_faces' in data:
        Globals.many_faces = bool(data['many_faces'])
    if 'mouth_mask' in data: # HTML ID is 'mouth-mask'
        Globals.mouth_mask = bool(data['mouth_mask']) # Maps to Globals.mouth_mask
    # Add more settings as they are defined in Globals and the UI
    if 'frame_processors' in data: # Example for a more complex setting
        Globals.frame_processors = data['frame_processors'] # Assuming it's a list of strings

    # A more generic way if keys match Globals attributes:
    # for key, value in data.items():
    #     if hasattr(Globals, key):
    #         # Be careful with types, e.g. ensuring booleans are booleans
    #         if isinstance(getattr(Globals, key, None), bool):
    #             setattr(Globals, key, bool(value))
    #         else:
    #             setattr(Globals, key, value)

    return jsonify({'message': 'Settings updated'}), 200

@app.route('/start_processing', methods=['POST'])
def start_processing():
    if not Globals.source_path or not os.path.exists(Globals.source_path):
        return jsonify({'error': 'Source path not set or invalid'}), 400
    if not Globals.target_path or not os.path.exists(Globals.target_path):
        return jsonify({'error': 'Target path not set or invalid'}), 400

    # Determine a unique output filename and set Globals.output_path
    target_filename = os.path.basename(Globals.target_path)
    filename, ext = os.path.splitext(target_filename)
    unique_output_filename = f"{filename}_processed_{int(time.time())}{ext}"
    Globals.output_path = os.path.join(app.config['PROCESSED_OUTPUTS_FOLDER'], unique_output_filename)

    # Ensure default frame processors are set if none are provided by the client
    if not Globals.frame_processors:
        Globals.frame_processors = ['face_swapper'] # Default to face_swapper
        print("Warning: No frame processors selected by client, defaulting to 'face_swapper'.")

    try:
        # Log current settings being used
        print(f"Preparing to process with core engine. Source: {Globals.source_path}, Target: {Globals.target_path}, Output: {Globals.output_path}")
        print(f"Options: Keep FPS: {Globals.keep_fps}, Keep Audio: {Globals.keep_audio}, Many Faces: {Globals.many_faces}")
        print(f"Frame Processors: {Globals.frame_processors}")
        # Ensure necessary resources are available and limited (e.g. memory)
        # This was part of the old core.run() sequence.
        # Consider if pre_check from core should be called here too, or if it's mainly for CLI
        # For now, webapp assumes inputs are valid if they exist.
        core.limit_resources()

        # Call the refactored core processing function
        processing_result = core.process_media()

        if processing_result.get('success'):
            final_output_path = processing_result.get('output_path', Globals.output_path) # Use path from result if available
            # Ensure the unique_output_filename matches the actual output from process_media if it changed it
            # For now, we assume process_media uses Globals.output_path as set above.
            print(f"Core processing successful. Output at: {final_output_path}")
            return jsonify({
                'message': 'Processing complete',
                'output_filename': os.path.basename(final_output_path),
                'download_url': f'/get_output/{os.path.basename(final_output_path)}'
            })
        else:
            print(f"Core processing failed: {processing_result.get('error')}")
            # If NSFW, include that info if process_media provides it
            if processing_result.get('nsfw'):
                 return jsonify({'error': processing_result.get('error', 'NSFW content detected.'), 'nsfw': True}), 400 # Bad request due to content
            return jsonify({'error': processing_result.get('error', 'Unknown error during processing')}), 500

    except Exception as e:
        # This is a fallback for unexpected errors not caught by core.process_media
        print(f"An unexpected error occurred in /start_processing endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An critical unexpected error occurred: {str(e)}'}), 500
    finally:
        # Always attempt to clean up temp files, regardless of success or failure
        # core.cleanup_temp_files() takes no args now for webapp context (quit_app=False is default)
        print("Executing cleanup of temporary files from webapp.")
        core.cleanup_temp_files()


@app.route('/get_output/<filename>')
def get_output(filename):
    return send_from_directory(app.config['PROCESSED_OUTPUTS_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    # Initialize any necessary globals or configurations from core logic if needed
    # For example, if core.parse_args() sets up initial globals from some defaults:
    # import modules.core as main_core
    # main_core.parse_args([]) # Pass empty list or appropriate defaults if it expects CLI args

    # For development, directly run the Flask app.
    # For production, a WSGI server like Gunicorn would be used.
    app.run(debug=True, host='0.0.0.0', port=5000)


# Video Feed Section
def generate_frames():
    global VIDEO_CAMERA
    global VIDEO_CAMERA, prev_time, frame_count, current_fps
    print("generate_frames: Attempting to open camera...")

    # Determine camera index (e.g., from Globals or default to 0)
    camera_index = 0 # Or Globals.camera_index if you add such a setting
    VIDEO_CAMERA = cv2.VideoCapture(camera_index)

    if not VIDEO_CAMERA.isOpened():
        print(f"Error: Could not open video camera at index {camera_index}.")
        # TODO: Yield a placeholder image with an error message
        return

    print("generate_frames: Camera opened. Initializing settings for live processing.")
    prev_time = time.time()
    frame_count = 0
    current_fps = 0

    source_face = None
    if Globals.source_path and not Globals.map_faces: # map_faces logic for live might be complex
        try:
            source_image_cv2 = cv2.imread(Globals.source_path)
            if source_image_cv2 is not None:
                source_face = get_one_face(source_image_cv2)
            if source_face is None:
                print("Warning: No face found in source image for live preview.")
        except Exception as e:
            print(f"Error loading source image for live preview: {e}")

    # Get frame processors
    # Ensure Globals.frame_processors is a list. If it can be None, default to an empty list.
    current_frame_processors = Globals.frame_processors if Globals.frame_processors is not None else []
    active_frame_processors = frame_processors_core.get_frame_processors_modules(current_frame_processors)

    # Example: Conditionally remove face enhancer if its toggle is off
    # This assumes fp_ui structure; adjust if it's different or not used for live mode.
    if not Globals.fp_ui.get('face_enhancer', False) and any(p.NAME == 'DLC.FACE-ENHANCER' for p in active_frame_processors):
        active_frame_processors = [p for p in active_frame_processors if p.NAME != 'DLC.FACE-ENHANCER']
        print("Live Preview: Face Enhancer disabled by UI toggle.")


    print(f"Live Preview: Active processors: {[p.NAME for p in active_frame_processors if hasattr(p, 'NAME')]}")

    try:
        while VIDEO_CAMERA and VIDEO_CAMERA.isOpened(): # Check if VIDEO_CAMERA is not None
            success, frame = VIDEO_CAMERA.read()
            if not success:
                print("Error: Failed to read frame from camera during live feed.")
                break

            processed_frame = frame.copy()

            if Globals.live_mirror:
                processed_frame = cv2.flip(processed_frame, 1)

            # Apply Processing
            # Apply Processing
            if Globals.map_faces:
                if Globals.simple_map: # Check if mappings are submitted and processed
                    for processor in active_frame_processors:
                        if hasattr(processor, 'process_frame_v2') and callable(processor.process_frame_v2):
                            try:
                                processed_frame = processor.process_frame_v2(processed_frame)
                            except Exception as e:
                                print(f"Error applying mapped processor {processor.NAME if hasattr(processor, 'NAME') else 'Unknown'} in live feed: {e}")
                                cv2.putText(processed_frame, "Error in mapped processing", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # else: No v2 method, map_faces might not apply or needs different handling
                else: # map_faces is true, but mappings not submitted/valid
                    cv2.putText(processed_frame, "Map Faces: Mappings not submitted or invalid.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif source_face: # Not map_faces, but single source face is available
                for processor in active_frame_processors:
                    try:
                        if hasattr(processor, 'process_frame') and callable(processor.process_frame):
                            if processor.NAME == 'DLC.FACE-ENHANCER':
                                processed_frame = processor.process_frame(None, processed_frame)
                            else:
                                processed_frame = processor.process_frame(source_face, processed_frame)
                    except Exception as e:
                        print(f"Error applying single source processor {processor.NAME if hasattr(processor, 'NAME') else 'Unknown'} in live feed: {e}")

            elif not Globals.source_path: # No map_faces and no single source image
                 cv2.putText(processed_frame, "No Source Image Selected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # FPS Calculation & Overlay
            if Globals.show_fps:
                frame_count += 1
                now = time.time()
                # Calculate FPS over a 1-second interval
                if (now - prev_time) > 1:
                    current_fps = frame_count / (now - prev_time)
                    prev_time = now
                    frame_count = 0

                cv2.putText(processed_frame, f"FPS: {current_fps:.2f}", (10, processed_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Encode the processed_frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                print("Error: Failed to encode processed frame to JPEG.")
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except GeneratorExit:
        print("generate_frames: Client disconnected.")
    except Exception as e:
        print(f"Exception in generate_frames main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("generate_frames: Releasing camera.")
        if VIDEO_CAMERA:
            VIDEO_CAMERA.release()
        VIDEO_CAMERA = None # Reset global camera object


@app.route('/video_feed')
def video_feed():
    print("Request received for /video_feed")
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Optional: Endpoint to explicitly stop the camera if needed.
# This is tricky with a global VIDEO_CAMERA and HTTP's stateless nature.
# A more robust solution might involve websockets or a different camera management strategy.
@app.route('/stop_video_feed', methods=['POST'])
def stop_video_feed():
    global VIDEO_CAMERA
    print("/stop_video_feed called")
    if VIDEO_CAMERA:
        print("Releasing video camera from /stop_video_feed")
        VIDEO_CAMERA.release()
        VIDEO_CAMERA = None
        return jsonify({'message': 'Video feed stopped.'})
    return jsonify({'message': 'No active video feed to stop.'})

@app.route('/get_target_faces_for_mapping', methods=['GET'])
def get_target_faces_for_mapping_route():
    global target_path_web # Use the web-specific target path
    if not target_path_web or not os.path.exists(target_path_web):
        return jsonify({'error': 'Target image not uploaded or path is invalid.'}), 400

    if not util_is_image(target_path_web): # Use the utility function for checking image type
        return jsonify({'error': 'Target file is not a valid image for face mapping.'}), 400

    try:
        # This function will populate Globals.source_target_map
        # It expects the target image path to be in Globals.target_path for its internal logic
        # So, ensure Globals.target_path is also set to target_path_web for this call
        # This is a bit of a workaround due to how get_unique_faces_from_target_image uses Globals
        original_global_target_path = Globals.target_path
        Globals.target_path = target_path_web

        get_unique_faces_from_target_image() # This should fill Globals.source_target_map

        # Restore original Globals.target_path if it was different (e.g. from a previous full processing run)
        # For web UI flow, target_path_web and Globals.target_path will typically be the same after an upload.
        Globals.target_path = original_global_target_path

        if not Globals.source_target_map:
            return jsonify({'error': 'No faces found in the target image or error during analysis.'}), 404

        response_data = []
        for item in Globals.source_target_map:
            target_cv2_img = item['target']['cv2']
            if target_cv2_img is None: # Should not happen if map is populated correctly
                continue

            _, buffer = cv2.imencode('.jpg', target_cv2_img)
            b64_img = base64.b64encode(buffer).decode('utf-8')
            response_data.append({'id': item['id'], 'image_b64': b64_img})

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in /get_target_faces_for_mapping: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/submit_face_mappings', methods=['POST'])
def submit_face_mappings_route():
    if 'target_ids_json' not in request.form:
        return jsonify({'error': 'No target_ids_json provided.'}), 400

    try:
        target_ids = json.loads(request.form['target_ids_json'])
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON in target_ids_json.'}), 400

    if not Globals.source_target_map:
        # This implies /get_target_faces_for_mapping was not called or failed.
        # Or, it could be cleared. Re-populate it if target_path_web is available.
        if target_path_web and os.path.exists(target_path_web) and util_is_image(target_path_web):
            print("Re-populating source_target_map as it was empty during submit.")
            original_global_target_path = Globals.target_path
            Globals.target_path = target_path_web
            get_unique_faces_from_target_image()
            Globals.target_path = original_global_target_path
            if not Globals.source_target_map:
                 return jsonify({'error': 'Could not re-initialize target faces. Please re-upload target image.'}), 500
        else:
            return jsonify({'error': 'Target face map not initialized. Please upload target image again.'}), 500


    all_mappings_valid = True
    processed_ids = set()

    for target_id_str in target_ids:
        target_id = int(target_id_str) # Ensure it's an integer if IDs are integers
        file_key = f'source_file_{target_id}'

        if file_key not in request.files:
            print(f"Warning: Source file for target_id {target_id} not found in submission.")
            # Mark this mapping as invalid or skip? For now, we require all submitted IDs to have files.
            # If a file is optional for a target, client should not include its ID in target_ids_json.
            # However, Globals.source_target_map will still have this target. We just won't assign a source to it.
            continue

        source_file = request.files[file_key]
        if source_file.filename == '':
            print(f"Warning: Empty filename for source file for target_id {target_id}.")
            continue # Skip if no file was actually selected for this input

        # Save the uploaded source file temporarily for this mapping
        temp_source_filename = f"temp_source_for_target_{target_id}_{secure_filename(source_file.filename)}"
        temp_source_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_source_filename)
        source_file.save(temp_source_filepath)

        source_cv2_img = cv2.imread(temp_source_filepath)
        if source_cv2_img is None:
            print(f"Error: Could not read saved source image for target_id {target_id} from {temp_source_filepath}")
            # all_mappings_valid = False # Decide if one bad source fails all
            # os.remove(temp_source_filepath) # Clean up
            continue # Skip this mapping

        source_face_obj = get_one_face(source_cv2_img) # This also returns the cropped face usually

        if source_face_obj:
            map_entry_found = False
            for map_item in Globals.source_target_map:
                if str(map_item['id']) == str(target_id): # Compare as strings or ensure IDs are consistent type
                    # The 'face' from get_one_face is the full Face object.
                    # The 'cv2' image from get_one_face is the cropped face.
                    # We need to store both, similar to how the original UI might have done.
                    # Let's assume get_one_face returns a tuple (Face_object, cropped_cv2_image)
                    # or that Face_object itself contains the cropped image if needed later.
                    # For now, storing the Face object which includes embedding and bbox.
                    # The cropped image can be re-derived or stored if `get_one_face` provides it.
                    # Let's assume `get_one_face` is just the Face object for simplicity here,
                    # and the cropped image for `source_target_map` needs to be handled.
                    # A better `get_one_face` might return a dict {'face': Face, 'cv2': cropped_img}

                    # Simplified: get_one_face returns the Face object, and we'll use that.
                    # The `ui.update_popup_source` implies the map needs {'cv2': cropped_img, 'face': Face_obj}
                    # Let's assume `source_face_obj` is the Face object. We need its cropped image.
                    # This might require a helper or for get_one_face to return it.
                    # For now, we'll store the Face object. The cropped image part for source_target_map
                    # might need adjustment based on face_analyser's exact return for get_one_face.
                    # A common pattern is that the Face object itself has bbox, and you can crop from original using that.

                    # Let's assume we need to manually crop based on the Face object from get_one_face
                    # This is a placeholder - exact cropping depends on what get_one_face returns and what processors need
                    # For now, we'll just store the Face object.
                    # If `face_swapper`'s `process_frame_v2` needs cropped source images in `source_target_map`,
                    # this part needs to ensure they are correctly populated.
                    # For simplicity, assuming `get_one_face` returns the main `Face` object, and `face_swapper` can use that.
                    # The `source_target_map` structure is critical.
                    # Looking at `face_swapper.py`, `process_frame_v2` uses `Globals.simple_map`.
                    # `simplify_maps()` populates `simple_map` from `source_target_map`.
                    # `simplify_maps()` expects `item['source']['face']` to be the source `Face` object.

                    map_item['source'] = {'face': source_face_obj, 'cv2': source_cv2_img} # Store the original uploaded source, not necessarily cropped yet. Processors handle cropping.
                    map_entry_found = True
                    processed_ids.add(target_id)
                    break

            if not map_entry_found:
                print(f"Warning: Target ID {target_id} from submission not found in existing map.")
                all_mappings_valid = False # Or handle as error
        else:
            print(f"Warning: No face found in uploaded source for target_id {target_id}.")
            # Mark this specific mapping as invalid by not adding a 'source' to it, or removing it.
            # For now, we just don't add a source. simplify_maps should handle items without a source.
            all_mappings_valid = False # if strict, one failed source makes all invalid for this submission batch

        # Clean up the temporary saved source file
        if os.path.exists(temp_source_filepath):
            os.remove(temp_source_filepath)

    # Clear 'source' for any target_ids that were in source_target_map but not in this submission
    # or if their source file didn't yield a face.
    for map_item in Globals.source_target_map:
        if map_item['id'] not in processed_ids and 'source' in map_item:
            del map_item['source']


    if not all_mappings_valid: # Or based on a stricter check
        # simplify_maps() will still run and create mappings for valid pairs
        print("simplify_maps: Some mappings may be invalid or incomplete.")

    simplify_maps() # Populate Globals.simple_map based on updated Globals.source_target_map

    # For debugging:
    # print("Updated source_target_map:", Globals.source_target_map)
    # print("Generated simple_map:", Globals.simple_map)

    if not Globals.simple_map and all_mappings_valid and target_ids: # If all submitted were meant to be valid but simple_map is empty
        return jsonify({'error': 'Mappings processed, but no valid face pairs were established. Check source images.'}), 400

    Globals.map_faces = True # Crucial: Set this global so processing functions know to use the map
    return jsonify({'message': 'Face mappings submitted and processed.'})

    # except Exception as e:
    #     print(f"Error in /submit_face_mappings: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
