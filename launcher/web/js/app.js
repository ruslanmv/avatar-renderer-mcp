// ============================================================================
// Avatar Renderer - Frontend Application Logic
// ============================================================================

// Global state
let config = null;
let isGeneratingAudio = false;
let isGeneratingVideo = false;

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize the application on page load
 */
async function initApp() {
    console.log('[INFO] Initializing application...');

    try {
        // Get configuration from Python backend
        config = await eel.get_config()();
        console.log('[INFO] Configuration loaded:', config);

        // Populate UI with configuration
        populateLanguages();
        populateVoices();
        populateQualityModes();
        updateSampleButtons();

        // Set default paths
        document.getElementById('avatar-path').value = config.default_avatar;
        document.getElementById('output-dir').value = config.output_dir;

        // Check API health
        checkAPIHealth();

        // Set up event listeners
        setupEventListeners();

        log('Application initialized successfully!', 'success');
    } catch (error) {
        console.error('[ERROR] Initialization failed:', error);
        log('Failed to initialize application: ' + error, 'error');
    }
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Language change - update sample buttons
    document.getElementById('language').addEventListener('change', updateSampleButtons);
}

// ============================================================================
// UI Population
// ============================================================================

/**
 * Populate language dropdown
 */
function populateLanguages() {
    const select = document.getElementById('language');
    select.innerHTML = '';

    for (const [code, name] of Object.entries(config.languages)) {
        const option = document.createElement('option');
        option.value = code;
        option.textContent = name;
        select.appendChild(option);
    }
}

/**
 * Populate voice profiles dropdown
 */
function populateVoices() {
    const select = document.getElementById('voice');
    select.innerHTML = '';

    for (const [key, profile] of Object.entries(config.voices)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = profile.name;
        select.appendChild(option);
    }
}

/**
 * Populate quality modes dropdown
 */
function populateQualityModes() {
    const select = document.getElementById('quality');
    select.innerHTML = '';

    for (const [key, name] of Object.entries(config.quality_modes)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = name;
        select.appendChild(option);
    }
}

/**
 * Update sample text buttons based on selected language
 */
function updateSampleButtons() {
    const language = document.getElementById('language').value;
    const container = document.getElementById('sample-buttons');
    container.innerHTML = '';

    const samples = config.samples[language] || config.samples['en'];

    for (const [key, text] of Object.entries(samples)) {
        const button = document.createElement('button');
        button.className = 'btn btn-secondary btn-sm';
        button.textContent = key.charAt(0).toUpperCase() + key.slice(1);
        button.onclick = () => loadSample(key);
        container.appendChild(button);
    }
}

// ============================================================================
// API Health Check
// ============================================================================

/**
 * Check backend API health status
 */
async function checkAPIHealth() {
    const statusBadge = document.getElementById('api-status');

    try {
        const result = await eel.check_api_health()();

        if (result.status === 'online') {
            statusBadge.textContent = 'API Connected';
            statusBadge.className = 'status-badge online';
        } else {
            statusBadge.textContent = 'API Offline';
            statusBadge.className = 'status-badge offline';
            log('Backend API is offline. Please start the server.', 'warning');
        }
    } catch (error) {
        statusBadge.textContent = 'API Error';
        statusBadge.className = 'status-badge offline';
        log('Failed to check API health: ' + error, 'error');
    }

    // Re-check every 10 seconds
    setTimeout(checkAPIHealth, 10000);
}

// ============================================================================
// Sample Text Loading
// ============================================================================

/**
 * Load a sample text into the textarea
 */
async function loadSample(sampleType) {
    const language = document.getElementById('language').value;
    const text = await eel.get_sample_text(language, sampleType)();

    if (text) {
        document.getElementById('text-input').value = text;
        log(`Loaded ${sampleType} sample text`, 'info');
    }
}

// ============================================================================
// Avatar Selection
// ============================================================================

/**
 * Handle avatar file selection
 */
async function handleAvatarSelect(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        const filePath = file.path || file.name;

        try {
            const result = await eel.set_avatar_path(filePath)();

            if (result.status === 'success') {
                document.getElementById('avatar-path').value = result.path;
                log('Avatar image selected: ' + file.name, 'success');
            } else {
                log('Failed to set avatar: ' + result.error, 'error');
            }
        } catch (error) {
            log('Error selecting avatar: ' + error, 'error');
        }
    }
}

// ============================================================================
// Audio Generation
// ============================================================================

/**
 * Generate audio from text
 */
async function generateAudio() {
    if (isGeneratingAudio) {
        log('Audio generation already in progress', 'warning');
        return;
    }

    const text = document.getElementById('text-input').value.trim();
    if (!text) {
        log('Please enter some text first', 'warning');
        return;
    }

    const language = document.getElementById('language').value;
    const voiceProfile = document.getElementById('voice').value;
    const button = document.getElementById('generate-audio-btn');

    isGeneratingAudio = true;
    button.disabled = true;
    button.textContent = '⏳ Generating Audio...';

    log('Generating audio...', 'info');

    try {
        const result = await eel.generate_audio(text, language, voiceProfile)();

        if (result.status === 'success') {
            log(result.message, 'success');

            // Show audio player
            const audioPlayer = document.getElementById('audio-player');
            const container = document.getElementById('audio-player-container');

            audioPlayer.src = result.audio_path;
            container.style.display = 'block';

            log('Audio ready! You can now generate the video.', 'success');
        } else {
            log('Audio generation failed: ' + result.error, 'error');
        }
    } catch (error) {
        log('Error generating audio: ' + error, 'error');
    } finally {
        isGeneratingAudio = false;
        button.disabled = false;
        button.textContent = '🎵 Generate Audio';
    }
}

// ============================================================================
// Video Generation
// ============================================================================

/**
 * Generate talking avatar video
 */
async function generateVideo() {
    if (isGeneratingVideo) {
        log('Video generation already in progress', 'warning');
        return;
    }

    const avatarPath = document.getElementById('avatar-path').value;
    const qualityMode = document.getElementById('quality').value;
    const button = document.getElementById('generate-video-btn');
    const progressContainer = document.getElementById('progress-container');

    isGeneratingVideo = true;
    button.disabled = true;
    button.textContent = '⏳ Rendering...';
    progressContainer.style.display = 'block';

    log('Starting video generation...', 'info');
    log('This may take a few minutes depending on quality settings', 'info');

    try {
        const result = await eel.generate_video(avatarPath, qualityMode)();

        if (result.status === 'success') {
            log('════════════════════════════════', 'success');
            log('✅ VIDEO GENERATED SUCCESSFULLY!', 'success');
            log('Output: ' + result.video_path, 'success');
            log('════════════════════════════════', 'success');

            // Show success notification
            showNotification('Video generated successfully! 🎉', 'success');
        } else {
            log('Video generation failed: ' + result.error, 'error');
            showNotification('Video generation failed', 'error');
        }
    } catch (error) {
        log('Error generating video: ' + error, 'error');
        showNotification('An error occurred', 'error');
    } finally {
        isGeneratingVideo = false;
        button.disabled = false;
        button.textContent = '🎬 START RENDER';
        progressContainer.style.display = 'none';
    }
}

// ============================================================================
// Status Updates (Called from Python)
// ============================================================================

/**
 * Update status text (called from Python backend)
 */
eel.expose(update_status);
function update_status(message) {
    log(message, 'info');
    document.getElementById('progress-text').textContent = message;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Open output folder in system file explorer
 */
function openOutputFolder() {
    const outputDir = document.getElementById('output-dir').value;
    log('Opening output folder: ' + outputDir, 'info');

    // Note: This functionality is limited in web environment
    // In a real implementation, you'd call a Python function to open the folder
    log('Please check: ' + outputDir, 'info');
}

/**
 * Add log entry to the log output
 */
function log(message, type = 'info') {
    const logOutput = document.getElementById('log-output');
    const logLine = document.createElement('p');

    logLine.className = `log-line ${type}`;
    logLine.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;

    logOutput.appendChild(logLine);
    logOutput.scrollTop = logOutput.scrollHeight;

    // Keep only last 100 log entries
    while (logOutput.children.length > 100) {
        logOutput.removeChild(logOutput.firstChild);
    }
}

/**
 * Show notification (simple implementation)
 */
function showNotification(message, type) {
    // For now, just log it
    // In a real implementation, you might use a toast library
    console.log(`[NOTIFICATION] ${type.toUpperCase()}: ${message}`);
}

// ============================================================================
// Initialize on DOM ready
// ============================================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
