let config = null;

async function initApp() {
    try {
        config = await eel.get_config()();
        
        populateSelect('language', config.languages);
        populateSelect('voice', config.voices, (k,v) => v.name);
        populateSelect('quality', config.quality_modes);
        
        // Defaults
        document.getElementById('language').value = 'en';
        document.getElementById('voice').value = 'sophia';
        document.getElementById('quality').value = 'real_time';
        
        if(config.default_avatar) {
            document.getElementById('avatar-path').value = config.default_avatar;
            // Force avatar preview update if file exists
            await eel.set_avatar_path(config.default_avatar)().then(res => {
                if(res.status === 'success') document.getElementById('avatar-preview-img').src = res.preview_url;
            });
        }

        updateSamples();
        checkHealth();
        
        document.getElementById('language').addEventListener('change', updateSamples);
        
    } catch (e) {
        console.error("Init failed:", e);
        alert("Failed to initialize. Is the backend running?");
    }
}

function populateSelect(id, data, textFn = (k,v) => v) {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    for(const [k,v] of Object.entries(data)) {
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = textFn(k,v);
        sel.appendChild(opt);
    }
}

async function updateSamples() {
    const lang = document.getElementById('language').value;
    const container = document.getElementById('sample-buttons');
    container.innerHTML = '';
    
    const samples = config.samples[lang] || config.samples['en'];
    for(const key of Object.keys(samples)) {
        const btn = document.createElement('button');
        btn.className = 'btn secondary btn-sm';
        btn.textContent = key.charAt(0).toUpperCase() + key.slice(1);
        btn.onclick = async () => {
            const text = await eel.get_sample_text(lang, key)();
            document.getElementById('text-input').value = text;
        };
        container.appendChild(btn);
    }
}

async function checkHealth() {
    const el = document.getElementById('api-status');
    try {
        const res = await eel.check_api_health()();
        if(res.status === 'online') {
            el.textContent = 'Online';
            el.className = 'badge online';
        } else {
            throw new Error(res.error);
        }
    } catch(e) {
        el.textContent = 'Offline';
        el.className = 'badge offline';
    }
    setTimeout(checkHealth, 5000);
}

async function handleAvatarSelect(input) {
    if(input.files[0]) {
        const path = input.files[0].path || input.files[0].name; // .path works in Electron/Eel usually
        // If browser doesn't give full path due to security, user must use the Browse button or type path
        if(!path) return; 

        const res = await eel.set_avatar_path(path)();
        if(res.status === 'success') {
            document.getElementById('avatar-path').value = res.path;
            document.getElementById('avatar-preview-img').src = res.preview_url;
        } else {
            alert("Error: " + res.error);
        }
    }
}

async function generateAudio() {
    const btn = document.getElementById('generate-audio-btn');
    const player = document.getElementById('audio-player');
    
    btn.disabled = true;
    btn.innerHTML = '⏳ Generating...';
    player.style.display = 'none';

    const text = document.getElementById('text-input').value;
    const lang = document.getElementById('language').value;
    const voice = document.getElementById('voice').value;

    const res = await eel.generate_audio(text, lang, voice)();
    
    btn.disabled = false;
    btn.innerHTML = '🔊 Generate Audio';

    if(res.status === 'success') {
        player.src = res.preview_url;
        player.style.display = 'block';
        player.play();
    } else {
        alert("Audio Generation Failed: " + res.error);
    }
}

async function generateVideo() {
    const btn = document.getElementById('generate-video-btn');
    const prog = document.getElementById('progress-container');
    const resultContainer = document.getElementById('video-result-container');
    const videoPlayer = document.getElementById('video-player');

    btn.disabled = true;
    prog.classList.remove('hidden');
    resultContainer.classList.add('hidden');
    
    // Simulate progress bar animation
    const fill = document.getElementById('progress-fill');
    fill.style.width = '20%';
    
    const avatar = document.getElementById('avatar-path').value;
    const quality = document.getElementById('quality').value;

    const res = await eel.generate_video(avatar, quality)();

    btn.disabled = false;
    prog.classList.add('hidden');
    fill.style.width = '0%';

    if(res.status === 'success') {
        resultContainer.classList.remove('hidden');
        videoPlayer.src = res.preview_url;
        videoPlayer.play();
    } else {
        alert("Render Failed: " + res.error);
    }
}

// OS Actions
async function openOutputFolder() { await eel.open_output_folder()(); }
async function openVideoFile() { await eel.open_video_file()(); }

// Expose status updater
eel.expose(update_status);
function update_status(msg) {
    document.getElementById('progress-text').textContent = msg;
    // visual feedback on progress bar
    document.getElementById('progress-fill').style.width = '60%'; 
}

window.addEventListener('DOMContentLoaded', initApp);