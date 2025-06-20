console.log("main.js loaded");

document.addEventListener('DOMContentLoaded', () => {
    // File Upload Elements
    const sourceFileInput = document.getElementById('source-file');
    const targetFileInput = document.getElementById('target-file');
    const sourcePreview = document.getElementById('source-preview');
    const targetPreviewImage = document.getElementById('target-preview-image');
    const targetPreviewVideo = document.getElementById('target-preview-video');

    // Settings Elements
    const keepFpsCheckbox = document.getElementById('keep-fps');
    const keepAudioCheckbox = document.getElementById('keep-audio');
    const manyFacesCheckbox = document.getElementById('many-faces'); // General many_faces
    const mapFacesCheckbox = document.getElementById('map-faces-checkbox'); // Specific for face mapping UI
    const mouthMaskCheckbox = document.getElementById('mouth-mask');
    // Add other settings elements here

    // Status Element
    const statusMessage = document.getElementById('status-message');

    // Action Elements
    const startProcessingButton = document.getElementById('start-processing');
    const livePreviewButton = document.getElementById('live-preview');
    const processedPreviewImage = document.getElementById('processed-preview');
    const outputArea = document.getElementById('output-area');
    const downloadLink = document.getElementById('download-link');

    // Face Mapper Elements
    const faceMapperContainer = document.getElementById('face-mapper-container');
    const faceMapperArea = document.getElementById('face-mapper-area');
    const submitFaceMappingsButton = document.getElementById('submit-face-mappings');
    const faceMapperStatus = document.getElementById('face-mapper-status');

    // WebApp state (mirroring some crucial Globals for UI logic)
    let webAppGlobals = {
        target_path_web: null, // Store the uploaded target file's path for UI checks
        source_target_map_from_backend: [], // To hold face data from /get_target_faces_for_mapping
        currentFaceMappings: [] // To store { target_id, target_image_b64, source_file, source_b64_preview }
    };

    // Initially hide output area and face mapper
    if(outputArea) outputArea.style.display = 'none';
    if(faceMapperContainer) faceMapperContainer.style.display = 'none';
    if(submitFaceMappingsButton) submitFaceMappingsButton.style.display = 'none';


    // Function to handle file preview (generic for source and target main previews)
    function previewFile(file, imagePreviewElement, videoPreviewElement) {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (file.type.startsWith('image/')) {
                imagePreviewElement.src = e.target.result;
                imagePreviewElement.style.display = 'block';
                if (videoPreviewElement) videoPreviewElement.style.display = 'none';
            } else if (file.type.startsWith('video/')) {
                if (videoPreviewElement) {
                    videoPreviewElement.src = e.target.result;
                    videoPreviewElement.style.display = 'block';
                }
                imagePreviewElement.style.display = 'none';
            }
        };
        reader.readAsDataURL(file);
    }

    // Source File Upload
    sourceFileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;

        previewFile(file, sourcePreview, null); // Source is always an image

        const formData = new FormData();
        formData.append('file', file);

        statusMessage.textContent = 'Uploading source...';
        fetch('/upload/source', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Source upload error:', data.error);
                statusMessage.textContent = `Error: ${data.error}`;
            } else {
                console.log('Source uploaded:', data);
                statusMessage.textContent = 'Source uploaded successfully.';
                // Optionally, use data.filepath if server sends a path to a served file
            }
        })
        .catch(error => {
            console.error('Fetch error for source upload:', error);
            statusMessage.textContent = 'Upload failed. Check console.';
        });
    });

    // Target File Upload
    targetFileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;

        previewFile(file, targetPreviewImage, targetPreviewVideo); // Show preview in main target area

        const formData = new FormData();
        formData.append('file', file);

        statusMessage.textContent = 'Uploading target...';
        fetch('/upload/target', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Target upload error:', data.error);
                statusMessage.textContent = `Error: ${data.error}`;
                webAppGlobals.target_path_web = null;
            } else {
                console.log('Target uploaded:', data);
                statusMessage.textContent = 'Target uploaded successfully.';
                webAppGlobals.target_path_web = data.filepath; // Store the path from backend
                // If map faces is checked, try to load faces
                if (mapFacesCheckbox && mapFacesCheckbox.checked) {
                    fetchAndDisplayTargetFaces();
                }
            }
        })
        .catch(error => {
            console.error('Fetch error for target upload:', error);
            statusMessage.textContent = 'Upload failed. Check console.';
            webAppGlobals.target_path_web = null;
        });
    });

    // Settings Update Logic
    function sendSettings() {
        const settings = {
            keep_fps: keepFpsCheckbox ? keepFpsCheckbox.checked : undefined,
            keep_audio: keepAudioCheckbox ? keepAudioCheckbox.checked : undefined,
            many_faces: manyFacesCheckbox ? manyFacesCheckbox.checked : undefined, // General many_faces
            map_faces: mapFacesCheckbox ? mapFacesCheckbox.checked : undefined, // map_faces for backend processing
            mouth_mask: mouthMaskCheckbox ? mouthMaskCheckbox.checked : undefined,
            // Add other settings here based on their IDs
        };
        // Clean undefined values
        Object.keys(settings).forEach(key => settings[key] === undefined && delete settings[key]);


        console.log('Sending settings:', settings);
        statusMessage.textContent = 'Updating settings...';
        fetch('/update_settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Settings update error:', data.error);
                statusMessage.textContent = `Error: ${data.error}`;
            } else {
                console.log('Settings updated:', data);
                statusMessage.textContent = 'Settings updated.';
            }
        })
        .catch(error => {
            console.error('Fetch error for settings update:', error);
            statusMessage.textContent = 'Settings update failed. Check console.';
        });
    }

    // Add event listeners to general settings checkboxes
    [keepFpsCheckbox, keepAudioCheckbox, manyFacesCheckbox, mouthMaskCheckbox].forEach(checkbox => {
        if (checkbox) {
            checkbox.addEventListener('change', sendSettings);
        }
    });
     // Special handling for mapFacesCheckbox as it affects UI and backend settings
    if (mapFacesCheckbox) {
        mapFacesCheckbox.addEventListener('change', () => {
            sendSettings(); // Update backend about the map_faces state for processing
            if (mapFacesCheckbox.checked && webAppGlobals.target_path_web) {
                faceMapperContainer.style.display = 'block';
                fetchAndDisplayTargetFaces();
            } else {
                if (faceMapperContainer) faceMapperContainer.style.display = 'none';
                if (faceMapperArea) faceMapperArea.innerHTML = ''; // Clear existing faces
                if (submitFaceMappingsButton) submitFaceMappingsButton.style.display = 'none';
                if (faceMapperStatus) faceMapperStatus.textContent = 'Upload a target image and check "Map Specific Faces" to begin.';
                webAppGlobals.currentFaceMappings = []; // Clear mappings
            }
        });
    }

    // Initial load of settings (optional, requires backend endpoint /get_settings)
    // fetch('/get_settings')
    // .then(response => response.json())
    // .then(settings => {
    //     keepFpsCheckbox.checked = settings.keep_fps || false;
    //     keepAudioCheckbox.checked = settings.keep_audio || false;
    //     manyFacesCheckbox.checked = settings.many_faces || false;
    //     mouthMaskCheckbox.checked = settings.mouth_mask || false;
    //     // set other checkboxes
    //     statusMessage.textContent = 'Settings loaded.';
    // })
    // .catch(error => {
    //     console.error('Error fetching initial settings:', error);
    //     statusMessage.textContent = 'Could not load initial settings.';
    // });

    // Function to fetch and display target faces for mapping
    function fetchAndDisplayTargetFaces() {
        if (!mapFacesCheckbox || !mapFacesCheckbox.checked || !webAppGlobals.target_path_web) {
            if (faceMapperStatus) faceMapperStatus.textContent = 'Target image not uploaded or "Map Specific Faces" not checked.';
            return;
        }

        if (faceMapperStatus) faceMapperStatus.textContent = "Loading target faces...";
        if (faceMapperContainer) faceMapperContainer.style.display = 'block'; // Show container while loading

        fetch('/get_target_faces_for_mapping')
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error || `HTTP error ${response.status}`) });
                }
                return response.json();
            })
            .then(targetFaces => {
                if (!faceMapperArea || !submitFaceMappingsButton || !faceMapperStatus) return;

                faceMapperArea.innerHTML = ''; // Clear previous faces
                webAppGlobals.currentFaceMappings = []; // Reset mappings

                if (targetFaces.error) {
                    faceMapperStatus.textContent = `Error: ${targetFaces.error}`;
                    submitFaceMappingsButton.style.display = 'none';
                    return;
                }
                if (targetFaces.length === 0) {
                    faceMapperStatus.textContent = "No faces found in the target image for mapping.";
                    submitFaceMappingsButton.style.display = 'none';
                    return;
                }

                targetFaces.forEach(face => {
                    const faceDiv = document.createElement('div');
                    faceDiv.className = 'face-map-item'; // For styling
                    faceDiv.style = "border:1px solid #ccc; padding:10px; text-align:center; margin-bottom:10px;";

                    faceDiv.innerHTML = `<p>Target ID: ${face.id}</p>`;

                    const imgEl = document.createElement('img');
                    imgEl.src = 'data:image/jpeg;base64,' + face.image_b64;
                    imgEl.style = "max-width:100px; max-height:100px; display:block; margin:auto;";
                    faceDiv.appendChild(imgEl);

                    const sourceInput = document.createElement('input');
                    sourceInput.type = 'file';
                    sourceInput.accept = 'image/*';
                    sourceInput.id = `source-for-target-${face.id}`;
                    sourceInput.dataset.targetId = face.id;
                    sourceInput.style = "margin-top:10px;";
                    faceDiv.appendChild(sourceInput);

                    const sourcePreview = document.createElement('img');
                    sourcePreview.id = `source-preview-for-target-${face.id}`;
                    sourcePreview.style = "max-width:80px; max-height:80px; display:none; margin-top:5px; margin:auto;";
                    faceDiv.appendChild(sourcePreview);

                    faceMapperArea.appendChild(faceDiv);

                    // Initialize this target face in our mapping array
                    webAppGlobals.currentFaceMappings.push({
                        target_id: face.id,
                        target_image_b64: face.image_b64,
                        source_file: null,
                        source_b64_preview: null // Will hold base64 for preview from file reader
                    });

                    // Add event listener for the file input
                    sourceInput.addEventListener('change', (event) => {
                        const file = event.target.files[0];
                        const targetId = event.target.dataset.targetId;
                        const mappingIndex = webAppGlobals.currentFaceMappings.findIndex(m => m.target_id == targetId);

                        if (file && mappingIndex !== -1) {
                            webAppGlobals.currentFaceMappings[mappingIndex].source_file = file;

                            // Preview for this source
                            const reader = new FileReader();
                            reader.onload = (e) => {
                                sourcePreview.src = e.target.result;
                                sourcePreview.style.display = 'block';
                                webAppGlobals.currentFaceMappings[mappingIndex].source_b64_preview = e.target.result;
                            };
                            reader.readAsDataURL(file);
                        } else if (mappingIndex !== -1) {
                            webAppGlobals.currentFaceMappings[mappingIndex].source_file = null;
                            webAppGlobals.currentFaceMappings[mappingIndex].source_b64_preview = null;
                            sourcePreview.src = '#';
                            sourcePreview.style.display = 'none';
                        }
                    });
                });

                submitFaceMappingsButton.style.display = 'block';
                faceMapperStatus.textContent = "Please select a source image for each target face.";
            })
            .catch(error => {
                console.error('Error fetching/displaying target faces:', error);
                if (faceMapperStatus) faceMapperStatus.textContent = `Error loading faces: ${error.message || 'Unknown error'}`;
                if (submitFaceMappingsButton) submitFaceMappingsButton.style.display = 'none';
            });
    }

    if (submitFaceMappingsButton) {
        submitFaceMappingsButton.addEventListener('click', (event) => {
            event.preventDefault(); // Prevent any default form submission behavior

            if (faceMapperStatus) faceMapperStatus.textContent = "Submitting mappings...";

            const formData = new FormData();
            const targetIdsWithSource = [];

            webAppGlobals.currentFaceMappings.forEach(mapping => {
                if (mapping.source_file) {
                    formData.append(`source_file_${mapping.target_id}`, mapping.source_file, mapping.source_file.name);
                    targetIdsWithSource.push(mapping.target_id);
                }
            });

            if (targetIdsWithSource.length === 0) {
                if (faceMapperStatus) faceMapperStatus.textContent = "No source images selected to map.";
                // Potentially clear backend maps if no sources are provided? Or backend handles this.
                // For now, we can choose to send an empty list, or not send at all.
                // Let's send an empty list to indicate an explicit "clear" or "submit with no new sources".
                // The backend will then call simplify_maps() which would clear simple_map.
            }

            formData.append('target_ids_json', JSON.stringify(targetIdsWithSource));

            fetch('/submit_face_mappings', {
                method: 'POST',
                body: formData // FormData will set Content-Type to multipart/form-data automatically
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error || `HTTP error ${response.status}`) });
                }
                return response.json();
            })
            .then(data => {
                console.log('Mappings submission response:', data);
                if (faceMapperStatus) faceMapperStatus.textContent = data.message || "Mappings submitted successfully.";
                // Optionally hide the face mapper container or update UI
                // For now, user can manually uncheck "Map Specific Faces" to hide it.
                // Or, if processing is started, it will also clear.
                // Consider if mapFacesCheckbox should be set to true in Globals on backend now.
                // The backend /submit_face_mappings sets Globals.map_faces = True.
                // We should ensure the checkbox reflects this state if it's not already.
                if (mapFacesCheckbox && !mapFacesCheckbox.checked && targetIdsWithSource.length > 0) {
                    // If user submitted mappings, but then unchecked "Map Faces" before submission finished,
                    // we might want to re-check it for them, or let sendSettings handle it.
                    // For simplicity, backend sets Globals.map_faces = true. UI should reflect this.
                    // mapFacesCheckbox.checked = true; // This might trigger its change event again.
                    // Better to let sendSettings in mapFacesCheckbox handler manage consistency.
                }
                 if (targetIdsWithSource.length > 0) {
                    statusMessage.textContent = "Face mappings ready. You can now start processing or live preview with these mappings.";
                }

            })
            .catch(error => {
                console.error('Error submitting face mappings:', error);
                if (faceMapperStatus) faceMapperStatus.textContent = `Error: ${error.message || 'Failed to submit mappings.'}`;
            });
        });
    }


    // Start Processing Logic
    if (startProcessingButton) {
        startProcessingButton.addEventListener('click', () => {
            // When starting processing, clear any live feed from the preview area
            if (processedPreviewImage) {
                processedPreviewImage.src = "#"; // Clear src
                processedPreviewImage.style.display = 'block'; // Or 'none' if you prefer to hide it
            }
            // Potentially call /stop_video_feed if live feed was active and using a global camera object that needs release
            // For now, just clearing the src is the main action.

            statusMessage.textContent = 'Processing... Please wait.';
            statusMessage.textContent = 'Processing... Please wait.';
            if(outputArea) outputArea.style.display = 'none'; // Hide previous output

            // Ensure settings are sent before starting, or rely on them being up-to-date
            // For simplicity, we assume settings are current from checkbox listeners.
            // Alternatively, call sendSettings() here and chain the fetch.

            fetch('/start_processing', {
                method: 'POST',
                // No body needed if settings are read from Globals on backend
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('Processing error:', data.error);
                    statusMessage.textContent = `Error: ${data.error}`;
                    if(outputArea) outputArea.style.display = 'none';
                } else {
                    console.log('Processing complete:', data);
                    statusMessage.textContent = 'Processing complete!';
                    if (downloadLink && data.download_url) {
                        downloadLink.href = data.download_url; // Backend provides full URL for download
                        downloadLink.textContent = `Download ${data.output_filename || 'processed file'}`;
                        if(outputArea) outputArea.style.display = 'block';
                    } else {
                         if(outputArea) outputArea.style.display = 'none';
                    }
                }
            })
            .catch(error => {
                console.error('Fetch error for start processing:', error);
                statusMessage.textContent = 'Processing request failed. Check console.';
                if(outputArea) outputArea.style.display = 'none';
            });
        });
    }

    // Live Preview Logic
    if (livePreviewButton && processedPreviewImage) {
        let isLiveFeedActive = false; // State to toggle button

        livePreviewButton.addEventListener('click', () => {
            if (!isLiveFeedActive) {
                processedPreviewImage.src = '/video_feed';
                processedPreviewImage.style.display = 'block'; // Make sure it's visible
                statusMessage.textContent = 'Live feed started. Navigate away or click "Stop Live Feed" to stop.';
                livePreviewButton.textContent = 'Stop Live Feed';
                isLiveFeedActive = true;
                if(outputArea) outputArea.style.display = 'none'; // Hide download area
            } else {
                // Stop the feed
                processedPreviewImage.src = '#'; // Clear the image source
                // Optionally, set a placeholder: processedPreviewImage.src = "placeholder.jpg";
                statusMessage.textContent = 'Live feed stopped.';
                livePreviewButton.textContent = 'Live Preview';
                isLiveFeedActive = false;

                // Inform the backend to release the camera, if the backend supports it
                // This is important if the camera is a shared global resource on the server.
                fetch('/stop_video_feed', { method: 'POST' })
                .then(response => response.json())
                .then(data => console.log('Stop video feed response:', data))
                .catch(error => console.error('Error stopping video feed:', error));
            }
        });
    }
});
